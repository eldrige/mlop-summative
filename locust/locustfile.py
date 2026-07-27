"""
Locust load test for the CheckMe BUSI classifier.

Simulates a flood of concurrent users hitting the model. The heaviest task
is ``/api/predict`` (a real image forward-pass); lighter status/metrics calls
are mixed in to resemble a live dashboard.

Interactive UI:
    locust -f locust/locustfile.py --host http://localhost:8080

Headless (what scripts/run_load_test.sh uses):
    locust -f locust/locustfile.py --host http://localhost:8080 \
           --headless -u 100 -r 20 -t 1m --csv results/run
"""
from pathlib import Path

from locust import HttpUser, between, task

# Pick a real test image to POST. Falls back to a generated one if data
# isn't mounted (e.g. inside a minimal container).
def _sample_image() -> bytes:
    root = Path(__file__).resolve().parents[1]
    for cand in (root / "data" / "test").rglob("*.png"):
        return cand.read_bytes()
    # Fallback: a tiny in-memory PNG so the test still runs.
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (127, 127, 127)).save(buf, "PNG")
    return buf.getvalue()


SAMPLE = _sample_image()


class ClassifierUser(HttpUser):
    """A user that mostly predicts, occasionally checks status/metrics."""
    wait_time = between(0.1, 0.5)

    @task(8)
    def predict(self):
        self.client.post(
            "/api/predict",
            files={"file": ("scan.png", SAMPLE, "image/png")},
            name="/api/predict",
        )

    @task(1)
    def status(self):
        self.client.get("/api/status", name="/api/status")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
