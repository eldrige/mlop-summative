"""Smoke tests for the preprocessing + prediction + API wiring.

Run:  pytest -q
These are lightweight (no training) and skip gracefully if the dataset or
trained model isn't present, so CI stays green on a fresh checkout.
"""
import io

import numpy as np
import pytest
from PIL import Image

from src import preprocessing as pp


def _fake_png(size=(300, 260), color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_classes_are_binary():
    assert pp.CLASS_NAMES == ("benign", "malignant")
    assert pp.CLASS_TO_IDX == {"benign": 0, "malignant": 1}


def test_preprocess_single_shape_and_range():
    arr = pp.preprocess_single(_fake_png())
    assert arr.shape == (1, pp.IMG_SIZE, pp.IMG_SIZE, 3)
    assert 0.0 <= arr.min() and arr.max() <= 255.0


def test_class_weights_favour_minority():
    labels = np.array([0] * 80 + [1] * 20)
    w = pp.compute_class_weights(labels)
    assert w[1] > w[0]                    # malignant weighted up


def test_api_imports_and_has_routes():
    import src.api as api

    paths = {r.path for r in api.app.routes}
    for p in ("/", "/health", "/api/status", "/api/predict",
              "/api/upload", "/api/retrain", "/api/visualizations"):
        assert p in paths


@pytest.mark.skipif(not (pp.DATA_DIR / "test").exists(),
                    reason="dataset not downloaded")
def test_list_images_finds_test_set():
    paths, labels = pp.list_images(pp.DATA_DIR / "test")
    assert len(paths) == len(labels) > 0
    assert set(np.unique(labels).astype(int)).issubset({0, 1})
