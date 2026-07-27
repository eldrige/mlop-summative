"""
model.py — build, train, evaluate and (re)train the MobileNetV2
benign-vs-malignant classifier.

Two-phase transfer learning, identical in spirit to the Intro-to-ML
notebook:
  Phase 1 — freeze the ImageNet backbone, train the new head.
  Phase 2 — unfreeze the top ~30 backbone layers, fine-tune at a low LR.

The public entry points are:
  * ``build_model()``  — a fresh compiled model.
  * ``train()``        — full pipeline from ``data/`` to a saved ``.keras``
                         file plus an evaluation ``metrics.json``. Used both
                         for the initial model and for API-triggered retrains.
  * ``evaluate()``     — every metric the assignment asks for on a test set.

Run standalone:  python -m src.model --epochs-head 20 --epochs-finetune 12
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from src import preprocessing as pp

MODELS_DIR = pp.REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "mobilenet_busi.keras"      # the served model
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR = pp.REPO_ROOT / "reports"                # evaluation figures


# ── Architecture ─────────────────────────────────────────────────────
def build_model(img_size: int = pp.IMG_SIZE, augment: bool = True):
    """MobileNetV2 transfer-learning head. Returns ``(model, backbone)``."""
    import tensorflow as tf

    inp = tf.keras.Input((img_size, img_size, 3))
    x = pp.make_augment()(inp) if augment else inp
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)   # -> [-1, 1]
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out, name="mobilenetv2_busi")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, backbone


def _callbacks(monitor: str = "val_auc"):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.5, patience=3, min_lr=1e-6,
        ),
    ]


# ── Evaluation ───────────────────────────────────────────────────────
def evaluate(model, test_ds, y_true, save_figures: bool = True) -> dict:
    """Compute the full metric suite on the held-out test set."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score,
                                 roc_curve)

    y_true = np.asarray(y_true).astype(int)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": int(len(y_true)),
        "class_names": list(pp.CLASS_NAMES),
    }

    if save_figures:
        _save_eval_figures(y_true, y_prob, y_pred, metrics)
    return metrics


def _save_eval_figures(y_true, y_prob, y_pred, metrics) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix.
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=pp.CLASS_NAMES, yticklabels=pp.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (acc={metrics['accuracy']:.2f})")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — malignant detection"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)


# ── Training / retraining ────────────────────────────────────────────
def train(data_dir=pp.DATA_DIR, epochs_head: int = 20, epochs_finetune: int = 12,
          out_path: Path = MODEL_PATH, version_tag: str | None = None) -> dict:
    """End-to-end: load ``data/`` → two-phase train → evaluate → save.

    Returns the metrics dict. Writes ``models/mobilenet_busi.keras``,
    ``models/metrics.json``, ``models/model_metadata.json`` and a
    timestamped versioned copy under ``models/versions/``.
    """
    import tensorflow as tf

    tf.random.set_seed(pp.SEED)
    train_ds, val_ds, test_ds, meta = pp.build_splits(data_dir)
    class_w = {int(k): float(v) for k, v in meta["class_weights"].items()}
    print(f"train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} "
          f"class_weights={class_w}")

    model, backbone = build_model(augment=True)

    print("Phase 1 — train head (backbone frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head,
              class_weight=class_w, callbacks=_callbacks(), verbose=2)

    print("Phase 2 — fine-tune top of backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune,
              class_weight=class_w, callbacks=_callbacks(), verbose=2)

    metrics = evaluate(model, test_ds, meta["test_labels"])
    print("Test metrics:", json.dumps({k: v for k, v in metrics.items()
                                       if isinstance(v, float)}, indent=2))

    # Persist model + sidecar files.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "versions").mkdir(exist_ok=True)
    out_path = Path(out_path)
    model.save(out_path)
    tag = version_tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(MODELS_DIR / "versions" / f"mobilenet_busi_{tag}.keras")

    metrics["trained_at"] = dt.datetime.now().isoformat(timespec="seconds")
    metrics["version"] = tag
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # Cache EDA payload so the cloud service can render charts without data.
    try:
        from src import insights
        insights.save_cache(data_dir)
    except Exception as e:  # noqa: BLE001
        print(f"(viz cache skipped: {e})")
    METADATA_PATH.write_text(json.dumps({
        "version": tag,
        "trained_at": metrics["trained_at"],
        "architecture": "MobileNetV2 transfer learning",
        "img_size": pp.IMG_SIZE,
        "classes": list(pp.CLASS_NAMES),
        "n_train": meta["n_train"], "n_val": meta["n_val"],
        "n_test": meta["n_test"],
    }, indent=2))
    print(f"Saved model -> {out_path}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=20)
    ap.add_argument("--epochs-finetune", type=int, default=12)
    ap.add_argument("--data-dir", default=str(pp.DATA_DIR))
    args = ap.parse_args()
    train(data_dir=args.data_dir, epochs_head=args.epochs_head,
          epochs_finetune=args.epochs_finetune)
