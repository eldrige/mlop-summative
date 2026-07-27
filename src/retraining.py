"""
retraining.py — staging of uploaded data and orchestration of retrain jobs.

Bulk-uploaded images land in ``data/uploads/<class>/``. They accumulate
there until a retrain is triggered — either manually (a button in the UI)
or automatically once the number of pending images crosses
``RETRAIN_THRESHOLD`` (the "trigger for retraining when the need arises").

A retrain:
  1. copies every pending upload into ``data/train/<class>/``,
  2. runs the two-phase training in ``model.train()``,
  3. hot-reloads the served model,
  4. clears the upload staging area.

Jobs run on a background thread so the API stays responsive; ``JOB`` exposes
live status to the ``/api/retrain/status`` endpoint.
"""
from __future__ import annotations

import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from src import preprocessing as pp

UPLOAD_DIR = pp.DATA_DIR / "uploads"
RETRAIN_THRESHOLD = 20          # pending images that auto-recommend a retrain


# ── Upload staging ───────────────────────────────────────────────────
def save_upload(image_bytes: bytes, cls: str, filename: str) -> Path:
    """Persist one uploaded image under its class in the staging area."""
    if cls not in pp.CLASS_NAMES:
        raise ValueError(f"class must be one of {pp.CLASS_NAMES}, got {cls!r}")
    dest_dir = UPLOAD_DIR / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem or "img"
    dest = dest_dir / f"{stem}_{uuid.uuid4().hex[:8]}.png"
    _to_png(image_bytes, dest)
    return dest


def _to_png(image_bytes: bytes, dest: Path) -> None:
    """Normalise any uploaded image to a 3-channel PNG on disk."""
    import io

    from PIL import Image

    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(dest, "PNG")


def pending_counts() -> dict[str, int]:
    """How many staged images await training, per class."""
    out = {}
    for cls in pp.CLASS_NAMES:
        d = UPLOAD_DIR / cls
        out[cls] = len(list(d.glob("*.png"))) if d.is_dir() else 0
    return out


def pending_total() -> int:
    return sum(pending_counts().values())


def retrain_recommended() -> bool:
    return pending_total() >= RETRAIN_THRESHOLD


def _merge_uploads_into_train() -> int:
    """Move staged uploads into the training set. Returns images moved."""
    moved = 0
    for cls in pp.CLASS_NAMES:
        src = UPLOAD_DIR / cls
        if not src.is_dir():
            continue
        dst = pp.DATA_DIR / "train" / cls
        dst.mkdir(parents=True, exist_ok=True)
        for img in src.glob("*.png"):
            shutil.move(str(img), dst / f"upload_{uuid.uuid4().hex[:10]}.png")
            moved += 1
    return moved


# ── Background job manager ───────────────────────────────────────────
class RetrainJob:
    """Tracks the state of the most recent / running retrain."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.status = "idle"          # idle | running | completed | failed
        self.job_id = None
        self.started_at = None
        self.finished_at = None
        self.message = ""
        self.images_added = 0
        self.metrics = None
        self.error = None

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "job_id": self.job_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "images_added": self.images_added,
            "metrics": self.metrics,
            "error": self.error,
        }

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    def start(self, epochs_head: int = 12, epochs_finetune: int = 8) -> dict:
        """Kick off a retrain on a daemon thread (no-op if already running)."""
        with self._lock:
            if self.is_running:
                return {"started": False, "reason": "a retrain is already running",
                        **self.snapshot()}
            self.reset()
            self.status = "running"
            self.job_id = uuid.uuid4().hex[:12]
            self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.message = "Retrain started."
        t = threading.Thread(target=self._run, args=(epochs_head, epochs_finetune),
                             daemon=True)
        t.start()
        return {"started": True, **self.snapshot()}

    def _run(self, epochs_head: int, epochs_finetune: int):
        try:
            self.images_added = _merge_uploads_into_train()
            self.message = (f"Merged {self.images_added} new images; "
                            "training MobileNetV2...")
            # Imported here so a missing TF at import time never breaks the API.
            from src import model as model_mod
            from src import prediction

            metrics = model_mod.train(
                epochs_head=epochs_head, epochs_finetune=epochs_finetune,
            )
            prediction.reload_model()          # hot-swap the served model
            self.metrics = {k: v for k, v in metrics.items()
                            if isinstance(v, (int, float, str, list))}
            self.status = "completed"
            self.message = "Retrain completed; new model is live."
        except Exception as e:                 # noqa: BLE001
            self.status = "failed"
            self.error = f"{e}\n{traceback.format_exc()}"
            self.message = "Retrain failed."
        finally:
            self.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")


# Shared singleton used by the API.
JOB = RetrainJob()
