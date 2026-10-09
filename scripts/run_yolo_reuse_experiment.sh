#!/usr/bin/env bash
# Matched output allocation experiment under an independent cloud deadline.
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
.venv-gpu/bin/python -u scripts/yolo_output_reuse.py --engine "$CPG_RESULTS/measured/yolo.engine" \
  --controls "$CPG_RESULTS/measured" --output "$CPG_RESULTS/output-reuse.json"
CPG_REUSE_TIMEOUT=$(.venv-gpu/bin/python - <<'PY'
import math,os,time
remaining=float(os.environ['CPG_LEASE_DEADLINE'])-time.time()
if not math.isfinite(remaining) or remaining<600:
    raise SystemExit('Require at least ten minutes before the independent deadline')
print(int(remaining)-300)
PY
)
timeout --signal=TERM --kill-after=30 "$CPG_REUSE_TIMEOUT" .venv-gpu/bin/python -u scripts/yolo_reuse_benchmark.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" --output "$CPG_RESULTS/reuse-benchmark"
.venv-gpu/bin/python scripts/summarize_yolo_reuse.py "$CPG_RESULTS/reuse-benchmark" --output "$CPG_RESULTS/reuse-verification.json"
