#!/usr/bin/env bash
# Validate the experimental async submission correctness under an independent lease.
set -euo pipefail
: "${CPG_REVISION:?Set the committed source revision}"
: "${CPG_RESULT_NAME:?Set a new iteration result name}"
: "${CPG_LEASE_DEADLINE:?Require the independent lease deadline}"
[[ "$CPG_REVISION" =~ ^[a-f0-9]{40}$ ]]
[[ "$CPG_RESULT_NAME" =~ ^iteration-[a-zA-Z0-9-]+$ ]]
cd /workspace/flex-vision
export CPG_RESULTS="benchmark-results/$CPG_RESULT_NAME"
mkdir -p "$CPG_RESULTS"
trap 'find "$CPG_RESULTS" -name "*.csv" -exec gzip {} \;' EXIT
bash scripts/run_yolo_experiment.sh
CPG_STREAM_TIMEOUT=$(.venv-gpu/bin/python - <<'PY'
import math,os,time
remaining=float(os.environ['CPG_LEASE_DEADLINE'])-time.time()
if not math.isfinite(remaining) or remaining<600:
    raise SystemExit('Require ten minutes before the independent deadline')
print(int(remaining)-300)
PY
)
timeout --signal=TERM --kill-after=30 "$CPG_STREAM_TIMEOUT" .venv-gpu/bin/python -u scripts/yolo_stream_ownership.py --async-submit \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" --output "$CPG_RESULTS/async-ownership.json"
.venv-gpu/bin/python scripts/verify_yolo_stream_ownership.py "$CPG_RESULTS/async-ownership.json" --mode async-submit \
  --output "$CPG_RESULTS/async-verification.json"
