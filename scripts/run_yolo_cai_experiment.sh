#!/usr/bin/env bash
# Independently bounded real-inference CAI ownership validation.
set -euo pipefail
: "${CPG_LEASE_DEADLINE:?Require independent cleanup deadline}"
CPG_YOLO_MODE=validation bash scripts/run_yolo_experiment.sh
cd /workspace/flex-vision
export CPG_RESULTS="benchmark-results/$CPG_RESULT_NAME"
CPG_CAI_TIMEOUT=$(.venv-gpu/bin/python - <<'PY'
import math,os,time
remaining=float(os.environ['CPG_LEASE_DEADLINE'])-time.time()
if not math.isfinite(remaining) or remaining<600:
    raise SystemExit('Require ten minutes before independent termination')
print(int(remaining)-300)
PY
)
timeout --signal=TERM --kill-after=30 "$CPG_CAI_TIMEOUT" .venv-gpu/bin/python -u scripts/yolo_cai.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" \
  --output "$CPG_RESULTS/cai.json"
.venv-gpu/bin/python scripts/verify_yolo_cai.py "$CPG_RESULTS/cai.json" --output "$CPG_RESULTS/cai-audit.json"
