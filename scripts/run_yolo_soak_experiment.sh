#!/usr/bin/env bash
# Execute only within an independently guarded GPU lease.
set -euo pipefail
: "${CPG_REVISION:?Set the committed source revision}"
: "${CPG_RESULT_NAME:?Set a new iteration result name}"
: "${CPG_LEASE_DEADLINE:?Require the independent lease deadline}"
[[ "$CPG_REVISION" =~ ^[a-f0-9]{40}$ ]]
[[ "$CPG_RESULT_NAME" =~ ^iteration-[a-zA-Z0-9-]+$ ]]
cd /workspace/flex-vision
export CPG_RESULTS="benchmark-results/$CPG_RESULT_NAME"
mkdir -p "$CPG_RESULTS"
# Compress partial evidence on failure too. The coordinator retains exit status.
trap 'find "$CPG_RESULTS" -name "*.csv" -exec gzip {} \;' EXIT
bash scripts/run_yolo_experiment.sh
 .venv-gpu/bin/python -u scripts/yolo_output_reuse.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" \
  --output "$CPG_RESULTS/output-reuse.json"
CPG_SOAK_TIMEOUT=$(.venv-gpu/bin/python - <<'PY'
import json,math,os,time
from pathlib import Path
remaining=float(os.environ['CPG_LEASE_DEADLINE'])-time.time()
if not math.isfinite(remaining) or remaining<2700:
    raise SystemExit('Insufficient lease time: require 45 minutes after preparation')
bound=int(remaining)-300
Path(os.environ['CPG_RESULTS'],'soak-preparation.json').write_text(json.dumps({
    'remaining_lease_seconds':remaining,'execution_timeout_seconds':bound,
    'export_reserve_seconds':300,'gpu_soak_status':'not_started'},indent=2)+'\n')
print(bound)
PY
)
timeout --signal=TERM --kill-after=30 "$CPG_SOAK_TIMEOUT" .venv-gpu/bin/python -u scripts/yolo_sustained.py \
  --soak --engine "$CPG_RESULTS/measured/yolo.engine" \
  --controls "$CPG_RESULTS/measured" --output "$CPG_RESULTS/soak"
.venv-gpu/bin/python scripts/summarize_yolo_soak.py "$CPG_RESULTS/soak" \
  --output "$CPG_RESULTS/soak-verification.json"
