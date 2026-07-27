"""
Data acquisition for the BUSI (Breast Ultrasound Images) dataset.

Downloads the full BUSI dataset, keeps only the `benign` and `malignant`
classes (the `normal` class is excluded to match the Intro-to-ML report),
removes exact-duplicate images, and writes a deterministic, stratified
80/20 train/test split into  data/train/<class>/  and  data/test/<class>/.

Sources (tried in order):
  1. A local zip passed via --zip (offline / manual Kaggle download).
  2. Public Hugging Face mirror of the full dataset (no auth required).
  3. Kaggle CLI, if credentials are configured.

Run:  python scripts/acquire_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CLASSES = ("benign", "malignant")          # 'normal' intentionally excluded
TEST_SIZE = 0.20
SEED = 42

HF_ZIP_URL = (
    "https://huggingface.co/datasets/gymprathap/"
    "Breast-Cancer-Ultrasound-Images-Dataset/resolve/main/"
    "Breast-Cancer-Ultrasound-Images-Dataset.zip"
)


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a tiny textual progress meter."""
    import urllib.request

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as r:  # noqa: S310 (trusted, https)
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done/1e6:6.1f} / {total/1e6:.1f} MB  ({pct:4.1f}%)",
                          end="", flush=True)
        print()


def _get_zip(args) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="busi_"))
    if args.zip:
        src = Path(args.zip).expanduser()
        if not src.exists():
            sys.exit(f"--zip path not found: {src}")
        return src
    # Try the Hugging Face mirror.
    zpath = tmp / "busi.zip"
    try:
        _download(HF_ZIP_URL, zpath)
        return zpath
    except Exception as e:  # noqa: BLE001
        print(f"HF mirror failed ({e}); trying Kaggle CLI...")
    # Fall back to Kaggle.
    rc = os.system(
        "kaggle datasets download -d aryashah2k/"
        f"breast-ultrasound-images-dataset -q -p {tmp}"
    )
    kz = next(tmp.glob("*.zip"), None)
    if rc != 0 or kz is None:
        sys.exit("Could not obtain the dataset. Pass a local zip via --zip PATH.")
    return kz


def _find_class_dir(root: Path, cls: str) -> Path | None:
    """BUSI mirrors nest class folders differently; locate `<cls>/` robustly."""
    for p in root.rglob(cls):
        if p.is_dir() and any(p.glob("*.png")):
            return p
    # Some mirrors use capitalised folder names.
    for p in root.rglob(cls.capitalize()):
        if p.is_dir() and any(p.glob("*.png")):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="Path to a locally downloaded BUSI zip.")
    args = ap.parse_args()

    zpath = _get_zip(args)
    extract = Path(tempfile.mkdtemp(prefix="busi_ex_"))
    print(f"Extracting {zpath} ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(extract)

    # Collect (path, class) for benign/malignant, excluding *_mask images.
    items: list[tuple[Path, str]] = []
    for cls in CLASSES:
        cdir = _find_class_dir(extract, cls)
        if cdir is None:
            sys.exit(f"Could not locate a '{cls}' folder inside the archive.")
        for img in sorted(cdir.glob("*.png")):
            if "mask" in img.name.lower():
                continue
            items.append((img, cls))
    print(f"Found {len(items)} labelled images (masks excluded).")

    # De-duplicate by MD5 of raw bytes (BUSI has a few exact dupes).
    seen: dict[str, Path] = {}
    unique: list[tuple[Path, str]] = []
    for img, cls in items:
        h = hashlib.md5(img.read_bytes()).hexdigest()  # noqa: S324 (dedup only)
        if h not in seen:
            seen[h] = img
            unique.append((img, cls))
    print(f"After de-duplication: {len(unique)} images.")

    # Deterministic stratified split.
    import random
    from collections import defaultdict

    by_cls: dict[str, list[Path]] = defaultdict(list)
    for img, cls in unique:
        by_cls[cls].append(img)

    rng = random.Random(SEED)
    if DATA.exists():
        shutil.rmtree(DATA)
    counts = {"train": {}, "test": {}}
    for cls, imgs in by_cls.items():
        imgs = sorted(imgs)
        rng.shuffle(imgs)
        n_test = round(len(imgs) * TEST_SIZE)
        test_imgs, train_imgs = imgs[:n_test], imgs[n_test:]
        for split, group in (("train", train_imgs), ("test", test_imgs)):
            out = DATA / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(group):
                shutil.copy(src, out / f"{cls}_{i:04d}.png")
            counts[split][cls] = len(group)

    print("\nDataset written to", DATA)
    for split in ("train", "test"):
        line = "  ".join(f"{c}={counts[split].get(c,0)}" for c in CLASSES)
        print(f"  {split:5}: {line}  (total {sum(counts[split].values())})")


if __name__ == "__main__":
    main()
