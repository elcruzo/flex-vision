#!/usr/bin/env bash
# Separate correctness gate and focused CAI complete-inference capture.
set -euo pipefail
: "${CPG_LEASE_DEADLINE:?Require independent cleanup deadline}"
bash scripts/run_yolo_cai_experiment.sh
cd /workspace/flex-vision
export CPG_RESULTS="benchmark-results/$CPG_RESULT_NAME"
CPG_TRACE_TIMEOUT=$(.venv-gpu/bin/python - <<'PY'
import math,os,time
remaining=float(os.environ['CPG_LEASE_DEADLINE'])-time.time()
if not math.isfinite(remaining) or remaining<600:
    raise SystemExit('Require ten minutes before independent termination')
print(int(remaining)-300)
PY
)
timeout --signal=TERM --kill-after=30 "$CPG_TRACE_TIMEOUT" bash -euo pipefail <<'INNER'
apt-get update -qq
apt-get install -y -qq nsight-systems-2025.3.2
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
  -o "$CPG_RESULTS/cai-trace" .venv-gpu/bin/python -u scripts/yolo_cai_trace.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" \
  --cai-report "$CPG_RESULTS/cai.json" --output "$CPG_RESULTS/cai-trace.json"
nsys export --type sqlite -o "$CPG_RESULTS/cai-trace.sqlite" "$CPG_RESULTS/cai-trace.nsys-rep"
.venv-gpu/bin/python scripts/sanitize_trace.py "$CPG_RESULTS/cai-trace.sqlite" \
  --output "$CPG_RESULTS/cai-trace-sanitized.sqlite" > "$CPG_RESULTS/cai-trace-sanitization.json"
.venv-gpu/bin/python scripts/verify_yolo_cai_trace.py "$CPG_RESULTS/cai-trace.json" \
  --cai-report "$CPG_RESULTS/cai.json" --trace "$CPG_RESULTS/cai-trace-sanitized.sqlite" \
  --output "$CPG_RESULTS/cai-trace-audit.json"
INNER
