"""
insights.py — EDA/feature data for the dashboard charts.

``build_visualizations()`` computes class balance, per-class mean intensity
and raw image-size stats directly from ``data/``. Because the full dataset
(~200 MB) is not shipped to the cloud, ``train()`` caches the result to
``models/viz_cache.json`` so the deployed service can still render real EDA
charts without the images present. The API prefers live data and falls back
to this cache.
"""
from __future__ import annotations

import json
from pathlib import Path

from src import preprocessing as pp

VIZ_CACHE_PATH = pp.REPO_ROOT / "models" / "viz_cache.json"


def build_visualizations(data_dir: Path | str = pp.DATA_DIR) -> dict:
    """Compute the visualization payload from images on disk."""
    import numpy as np
    from PIL import Image

    data_dir = Path(data_dir)
    train_dist = pp.class_distribution(data_dir / "train")
    test_dist = pp.class_distribution(data_dir / "test")
    tr_paths, _ = pp.list_images(data_dir / "train")
    size_stats = pp.image_size_stats(tr_paths[:200]) if tr_paths else {}

    intensity: dict[str, float] = {}
    for cls in pp.CLASS_NAMES:
        vals = []
        for p in (data_dir / "train" / cls).glob("*.png"):
            if len(vals) >= 150:
                break
            vals.append(float(np.asarray(Image.open(p).convert("L")).mean()))
        intensity[cls] = sum(vals) / len(vals) if vals else 0.0

    return {
        "class_distribution": {"train": train_dist, "test": test_dist},
        "image_size_stats": size_stats,
        "intensity_by_class": intensity,
    }


def save_cache(data_dir: Path | str = pp.DATA_DIR) -> dict:
    """Compute and persist the viz payload (called after training)."""
    payload = build_visualizations(data_dir)
    VIZ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIZ_CACHE_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def get_visualizations(data_dir: Path | str = pp.DATA_DIR) -> dict:
    """Live data if available, otherwise the cached snapshot."""
    tr_paths, _ = pp.list_images(Path(data_dir) / "train")
    if tr_paths:
        return build_visualizations(data_dir)
    if VIZ_CACHE_PATH.exists():
        payload = json.loads(VIZ_CACHE_PATH.read_text())
        payload["_source"] = "cache"
        return payload
    return {"class_distribution": {"train": {}, "test": {}},
            "image_size_stats": {}, "intensity_by_class": {}}
