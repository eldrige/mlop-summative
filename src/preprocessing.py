"""
preprocessing.py — data loading, cleaning and tf.data pipelines for the
BUSI benign-vs-malignant classifier.

This is the single source of truth for how images are turned into model
input. It is imported by the training/retraining code (``model.py``), the
inference path (``prediction.py``) and the notebook, so every stage sees
exactly the same pixels.

Design mirrors the Intro-to-ML notebook:
  * grayscale ultrasound PNGs are read as 3 channels (ImageNet backbones
    expect RGB),
  * resized to 224×224 and left in [0, 255] — the model rescales to
    [-1, 1] internally,
  * horizontal flip / small rotation / zoom / contrast augmentation is
    applied on the training stream only.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

# ── Configuration (shared everywhere) ────────────────────────────────
IMG_SIZE = 224                       # MobileNetV2 native input
BATCH = 32
SEED = 42
CLASS_NAMES = ("benign", "malignant")          # index 0, 1 — 'normal' excluded
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


# ── Filesystem helpers ───────────────────────────────────────────────
def list_images(split_dir: Path | str) -> tuple[list[str], np.ndarray]:
    """Return ``(paths, labels)`` for one split directory.

    Expects ``<split_dir>/<class_name>/*.png``. Files whose name contains
    ``mask`` are ignored so segmentation masks never leak into training.
    """
    split_dir = Path(split_dir)
    paths: list[str] = []
    labels: list[int] = []
    for cls in CLASS_NAMES:
        cdir = split_dir / cls
        if not cdir.is_dir():
            continue
        for img in sorted(cdir.glob("*.png")):
            if "mask" in img.name.lower():
                continue
            paths.append(str(img))
            labels.append(CLASS_TO_IDX[cls])
    return paths, np.array(labels, dtype="float32")


def class_distribution(split_dir: Path | str) -> dict[str, int]:
    """Count images per class in a split (used by EDA + the dashboard)."""
    _, labels = list_images(split_dir)
    counts = Counter(labels.astype(int).tolist())
    return {c: int(counts.get(i, 0)) for i, c in enumerate(CLASS_NAMES)}


def image_size_stats(paths: Iterable[str]) -> dict[str, float]:
    """Width/height statistics of the raw images — motivates the resize."""
    from PIL import Image

    sizes = np.array([Image.open(p).size for p in paths])  # (N, 2): W, H
    if len(sizes) == 0:
        return {}
    return {
        "count": int(len(sizes)),
        "min_w": int(sizes[:, 0].min()), "max_w": int(sizes[:, 0].max()),
        "min_h": int(sizes[:, 1].min()), "max_h": int(sizes[:, 1].max()),
        "mean_w": float(sizes[:, 0].mean()), "mean_h": float(sizes[:, 1].mean()),
    }


def find_duplicates(paths: Iterable[str]) -> list[list[str]]:
    """Group paths whose raw bytes are identical (MD5). Data-quality check."""
    groups: dict[str, list[str]] = {}
    for p in paths:
        h = hashlib.md5(Path(p).read_bytes()).hexdigest()  # noqa: S324
        groups.setdefault(h, []).append(p)
    return [g for g in groups.values() if len(g) > 1]


# ── tf.data pipeline (imported lazily so non-TF callers stay light) ──
def _decode_factory(img_size: int):
    import tensorflow as tf

    def decode(path, label):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.cast(tf.image.resize(img, [img_size, img_size]), tf.float32)
        return img, label                      # stays in [0, 255]

    return decode


def make_augment():
    """Fresh augmentation block (Keras layers). One per model instance."""
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augment",
    )


def make_dataset(paths, labels, training: bool = False,
                 batch: int = BATCH, img_size: int = IMG_SIZE):
    """Build a batched, prefetched ``tf.data.Dataset`` from paths+labels."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = (
        ds.map(_decode_factory(img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights so the minority (malignant) class matters."""
    labels = np.asarray(labels).astype(int)
    counts = Counter(labels.tolist())
    n0 = counts.get(0, 0) or 1
    n1 = counts.get(1, 0) or 1
    return {0: 1.0, 1: float(n0 / n1)}


def build_splits(data_dir: Path | str = DATA_DIR, val_fraction: float = 0.15):
    """Return train/val/test tf.data datasets + metadata from disk.

    ``data_dir`` must contain ``train/`` and ``test/`` class folders (as
    produced by ``scripts/acquire_data.py``). A validation slice is carved
    out of the training portion, stratified and reproducibly.
    """
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir)
    tr_paths, tr_labels = list_images(data_dir / "train")
    te_paths, te_labels = list_images(data_dir / "test")
    if not tr_paths:
        raise FileNotFoundError(
            f"No training images under {data_dir/'train'}. "
            "Run: python scripts/acquire_data.py"
        )

    p_tr, p_val, y_tr, y_val = train_test_split(
        tr_paths, tr_labels, test_size=val_fraction,
        stratify=tr_labels, random_state=SEED,
    )

    meta = {
        "n_train": len(p_tr), "n_val": len(p_val), "n_test": len(te_paths),
        "class_names": list(CLASS_NAMES),
        "class_weights": compute_class_weights(y_tr),
        "test_paths": te_paths, "test_labels": te_labels.tolist(),
    }
    return (
        make_dataset(p_tr, y_tr, training=True),
        make_dataset(p_val, y_val),
        make_dataset(te_paths, te_labels),
        meta,
    )


def preprocess_single(image_bytes: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """Turn raw uploaded image bytes into a (1, H, W, 3) float array in [0,255].

    Accepts PNG/JPEG of any size/mode; grayscale is expanded to RGB. Used by
    the prediction endpoint so a single upload goes through the same resize
    as training.
    """
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype="float32")
    return arr[None, ...]
