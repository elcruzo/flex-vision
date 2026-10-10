#!/usr/bin/env bash
set -euo pipefail
: "${CPG_LEASE_DEADLINE:?Require independent cleanup deadline}"
CPG_YOLO_MODE=validation bash scripts/run_yolo_experiment.sh
cd /workspace/flex-vision
export CPG_RESULTS="benchmark-results/$CPG_RESULT_NAME"
CPG_STENCIL_TIMEOUT=$(.venv-gpu/bin/python - <<'PY'
import math,os,time
remaining=float(os.environ['CPG_LEASE_DEADLINE'])-time.time()
if not math.isfinite(remaining) or remaining<600:
    raise SystemExit('Require ten minutes before independent termination')
print(int(remaining)-300)
PY
)
timeout --signal=TERM --kill-after=30 "$CPG_STENCIL_TIMEOUT" .venv-gpu/bin/python -u scripts/yolo_stencil.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --output "$CPG_RESULTS/stencil.json"
.venv-gpu/bin/python scripts/verify_yolo_stencil.py "$CPG_RESULTS/stencil.json" --output "$CPG_RESULTS/stencil-audit.json"
