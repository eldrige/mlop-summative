# ─────────────────────────────────────────────────────────────
# CheckMe · BUSI benign-vs-malignant classifier API
# Single-stage image: FastAPI + TensorFlow-CPU serving MobileNetV2.
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1

WORKDIR /app

# System libs Pillow / TF need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# App code + the trained model + templates/assets.
# The 200MB dataset is NOT shipped: serving needs only the model, and the
# dashboard reads models/viz_cache.json for its EDA charts. Retraining in the
# cloud runs on images uploaded through the UI.
COPY src/ ./src/
COPY ui/ ./ui/
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

# Render provides $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
