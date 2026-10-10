#!/usr/bin/env bash
# Run focused async diagnostics after numerical and pending-consumer controls.
set -euo pipefail
: "${CPG_LEASE_DEADLINE:?Require independent termination}"
bash scripts/run_yolo_delayed_experiment.sh
cd /workspace/flex-vision
export CPG_RESULTS="benchmark-results/$CPG_RESULT_NAME"
CPG_DIAGNOSTIC_TIMEOUT=$(.venv-gpu/bin/python - <<'PY'
import math, os, time
remaining=float(os.environ['CPG_LEASE_DEADLINE'])-time.time()
if not math.isfinite(remaining) or remaining < 600:
    raise SystemExit('Require ten minutes before the independent deadline')
print(int(remaining)-300)
PY
)
export CPG_RESULTS
# The timeout covers installation, capture, sanitization, and verification.
timeout --signal=TERM --kill-after=30 "$CPG_DIAGNOSTIC_TIMEOUT" bash -euo pipefail <<'INNER'
apt-get update -qq
apt-get install -y -qq nsight-systems-2025.3.2
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
  -o "$CPG_RESULTS/async-trace" .venv-gpu/bin/python -u scripts/yolo_async_benchmark.py \
  --diagnostics --engine "$CPG_RESULTS/measured/yolo.engine" \
  --controls "$CPG_RESULTS/measured" --delayed-report "$CPG_RESULTS/delayed-consumer.json" \
  --output "$CPG_RESULTS/async-diagnostics"
nsys export --type sqlite -o "$CPG_RESULTS/async-trace.sqlite" "$CPG_RESULTS/async-trace.nsys-rep"
.venv-gpu/bin/python scripts/sanitize_trace.py "$CPG_RESULTS/async-trace.sqlite" \
  --output "$CPG_RESULTS/async-trace-sanitized.sqlite" > "$CPG_RESULTS/async-trace-sanitization.json"
.venv-gpu/bin/python scripts/verify_yolo_async_diagnostics.py "$CPG_RESULTS/async-diagnostics" \
  --trace "$CPG_RESULTS/async-trace-sanitized.sqlite" \
  --delayed-report "$CPG_RESULTS/delayed-consumer.json" \
  --output "$CPG_RESULTS/async-diagnostics/audit.json"
gzip "$CPG_RESULTS/async-diagnostics/samples.csv"
INNER
