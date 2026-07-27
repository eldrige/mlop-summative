"""
api.py — FastAPI service for the BUSI benign-vs-malignant classifier.

Exposes both a JSON API and a server-rendered dashboard:

  GET  /                     dashboard UI
  GET  /health               liveness probe (for Render / load balancer)
  GET  /api/status           uptime, model version, readiness, pending uploads
  GET  /api/metrics          latest held-out evaluation metrics
  GET  /api/visualizations   class balance + image-size stats for charts
  POST /api/predict          classify one uploaded image
  POST /api/upload           stage bulk images for retraining
  POST /api/retrain          trigger a retrain (background)
  GET  /api/retrain/status   poll the running/last retrain job

Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import insights, preprocessing as pp
from src import prediction, retraining
from src.model import METRICS_PATH, REPORTS_DIR

REPO = pp.REPO_ROOT
UI_DIR = REPO / "ui"
START_TIME = time.time()
MAX_UPLOAD_MB = 10

app = FastAPI(title="CheckMe — Breast Ultrasound Classifier",
              version="1.0.0",
              description="MobileNetV2 benign-vs-malignant BUSI classifier + MLOps pipeline")

templates = Jinja2Templates(directory=str(UI_DIR / "templates"))
if (UI_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")
if REPORTS_DIR.is_dir():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


# ── helpers ──────────────────────────────────────────────────────────
def _uptime() -> dict:
    secs = int(time.time() - START_TIME)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return {"seconds": secs, "human": f"{h}h {m}m {s}s"}


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


# ── UI ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── health / status ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "uptime": _uptime()["human"]}


@app.get("/api/status")
def status():
    return {
        "service": "up",
        "uptime": _uptime(),
        "model_ready": prediction.model_ready(),
        "model": prediction.model_metadata() if prediction.model_ready() else {},
        "pending_uploads": retraining.pending_counts(),
        "pending_total": retraining.pending_total(),
        "retrain_recommended": retraining.retrain_recommended(),
        "retrain_threshold": retraining.RETRAIN_THRESHOLD,
        "retrain_job": retraining.JOB.snapshot(),
    }


@app.get("/api/metrics")
def metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return JSONResponse({"detail": "No metrics yet — train the model first."}, 404)


@app.get("/api/visualizations")
def visualizations():
    """Feature/EDA data the dashboard turns into charts.

    Uses live images when present, else the cache written at train time.
    """
    payload = insights.get_visualizations()
    payload["figures"] = {
        "confusion_matrix": "/reports/confusion_matrix.png",
        "roc_curve": "/reports/roc_curve.png",
    }
    return payload


# ── prediction ───────────────────────────────────────────────────────
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await _read_upload(file)
    try:
        result = prediction.predict_image(data)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not process image: {e}")
    return result


# ── bulk upload + retraining ─────────────────────────────────────────
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...),
                 label: str = Form(...)):
    if label not in pp.CLASS_NAMES:
        raise HTTPException(422, f"label must be one of {list(pp.CLASS_NAMES)}")
    saved = 0
    for f in files:
        data = await _read_upload(f)
        try:
            retraining.save_upload(data, label, f.filename or "upload")
            saved += 1
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Failed on {f.filename}: {e}")
    return {
        "saved": saved,
        "label": label,
        "pending_total": retraining.pending_total(),
        "pending_counts": retraining.pending_counts(),
        "retrain_recommended": retraining.retrain_recommended(),
    }


@app.post("/api/retrain")
def retrain(epochs_head: int = Form(12), epochs_finetune: int = Form(8)):
    if retraining.pending_total() == 0 and not retraining.JOB.is_running:
        # Still allowed (retrain on existing data), but flag it.
        pass
    result = retraining.JOB.start(epochs_head=epochs_head,
                                  epochs_finetune=epochs_finetune)
    code = 202 if result.get("started") else 409
    return JSONResponse(result, status_code=code)


@app.get("/api/retrain/status")
def retrain_status():
    return retraining.JOB.snapshot()
