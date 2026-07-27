"""
prediction.py — load the trained model once and classify single images.

The model is loaded lazily and cached in a module-level ``_Predictor`` so the
API pays the (slow) load cost only on the first request, and every subsequent
prediction is a cheap forward pass. ``reload()`` swaps in a freshly retrained
model without restarting the server.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from src import preprocessing as pp
from src.model import METADATA_PATH, MODEL_PATH


class _Predictor:
    """Thread-safe holder for the Keras model + its metadata."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self._model = None
        self._meta: dict = {}
        self._lock = threading.Lock()

    # -- loading -------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:               # double-checked
                    self._load()

    def _load(self):
        import tensorflow as tf

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No model at {self.model_path}. Train first: "
                "python -m src.model"
            )
        self._model = tf.keras.models.load_model(self.model_path)
        if METADATA_PATH.exists():
            self._meta = json.loads(METADATA_PATH.read_text())

    def reload(self):
        """Force a reload after a retrain (called by the /retrain endpoint)."""
        with self._lock:
            self._model = None
            self._meta = {}
        self._ensure_loaded()

    @property
    def is_ready(self) -> bool:
        return self._model is not None or self.model_path.exists()

    @property
    def metadata(self) -> dict:
        self._ensure_loaded()
        return self._meta

    # -- inference -----------------------------------------------------
    def predict_bytes(self, image_bytes: bytes) -> dict:
        """Classify a single raw image (PNG/JPEG bytes).

        Returns the predicted class, the malignant probability, a per-class
        probability map and a confidence score.
        """
        self._ensure_loaded()
        arr = pp.preprocess_single(image_bytes)          # (1, 224, 224, 3)
        prob_malignant = float(self._model.predict(arr, verbose=0).ravel()[0])
        idx = int(prob_malignant >= 0.5)
        label = pp.CLASS_NAMES[idx]
        return {
            "prediction": label,
            "predicted_index": idx,
            "probabilities": {
                pp.CLASS_NAMES[0]: 1.0 - prob_malignant,
                pp.CLASS_NAMES[1]: prob_malignant,
            },
            "malignant_probability": prob_malignant,
            "confidence": max(prob_malignant, 1.0 - prob_malignant),
            "model_version": self._meta.get("version", "unknown"),
        }


# Module-level singleton used by the API.
_predictor = _Predictor()


def predict_image(image_bytes: bytes) -> dict:
    """Convenience wrapper around the shared predictor."""
    return _predictor.predict_bytes(image_bytes)


def reload_model() -> None:
    _predictor.reload()


def model_ready() -> bool:
    return _predictor.is_ready


def model_metadata() -> dict:
    return _predictor.metadata


if __name__ == "__main__":  # quick manual test: python -m src.prediction <img>
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.prediction <image_path>")
        raise SystemExit(1)
    out = predict_image(Path(sys.argv[1]).read_bytes())
    print(json.dumps(out, indent=2))
