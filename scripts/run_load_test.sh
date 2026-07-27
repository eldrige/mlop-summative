#!/usr/bin/env bash
# Flood-request simulation across different Docker container counts.
#
# For each replica count it (re)scales the stack, warms the model, runs a
# fixed Locust load, and saves per-run CSVs under results/. Compare the
# *_stats.csv files to see how latency improves as containers are added.
#
#   ./scripts/run_load_test.sh "1 2 3"     # test 1, 2 and 3 containers
#
set -euo pipefail
cd "$(dirname "$0")/.."

COUNTS=${1:-"1 2 3"}
USERS=${USERS:-100}
SPAWN=${SPAWN:-20}
DURATION=${DURATION:-1m}
HOST="http://localhost:8080"

mkdir -p results

command -v locust >/dev/null || { echo "Install locust: pip install locust"; exit 1; }

for N in $COUNTS; do
  echo "══════════════════════════════════════════════════════"
  echo "  Scaling to $N API container(s)"
  echo "══════════════════════════════════════════════════════"
  docker compose up -d --build --scale api="$N" api nginx

  echo "Warming up (waiting for model load) ..."
  for _ in $(seq 1 30); do
    if curl -fs "$HOST/health" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  # A few priming predictions so the first-request model load isn't counted.
  for _ in 1 2 3; do
    curl -fs -X POST "$HOST/api/predict" \
      -F "file=@$(find data/test -name '*.png' | head -1)" >/dev/null || true
  done

  echo "Running Locust: $USERS users, spawn $SPAWN/s, for $DURATION ..."
  locust -f locust/locustfile.py --host "$HOST" \
         --headless -u "$USERS" -r "$SPAWN" -t "$DURATION" \
         --csv "results/containers_${N}" --only-summary

  echo "Saved results/containers_${N}_stats.csv"
done

docker compose down
echo
echo "Done. Summary:"
for N in $COUNTS; do
  f="results/containers_${N}_stats.csv"
  [ -f "$f" ] && echo "  $N container(s): $(tail -1 "$f")"
done
