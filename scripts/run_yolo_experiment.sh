#!/usr/bin/env bash
# Run inside the disposable GPU host. The external controller owns expiration.
set -euo pipefail
: "${CPG_REVISION:?Set the committed source revision}"
: "${CPG_RESULT_NAME:?Set a new iteration result name}"
[[ "$CPG_REVISION" =~ ^[a-f0-9]{40}$ ]]
[[ "$CPG_RESULT_NAME" =~ ^iteration-[a-zA-Z0-9-]+$ ]]
cd /workspace/flex-vision
export CPG_RESULTS="benchmark-results/$CPG_RESULT_NAME"
mkdir -p "$CPG_RESULTS"
git checkout "$CPG_REVISION"
python -m venv --system-site-packages .venv-gpu
printf '\n.venv-gpu/\n' >> .git/info/exclude
.venv-gpu/bin/python -m pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv-gpu/bin/python -m pip install -e . -r requirements-yolo.txt cupy-cuda13x==14.2.0 cvcuda-cu13==0.18.0 tensorrt-cu13==10.13.3.9.post1 'cuda-toolkit[cudart]==13.0.0'
mkdir -p .cache/models .cache/ultralytics
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
curl -fL --retry 2 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -o .cache/models/yolov8n.pt
.venv-gpu/bin/python -m pip freeze > "$CPG_RESULTS/packages.txt"
.venv-gpu/bin/python -m pip check > "$CPG_RESULTS/pip-check.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$CPG_RESULTS/gpu.txt"
.venv-gpu/bin/python scripts/gpu_preflight.py --output "$CPG_RESULTS/preflight.json"
.venv-gpu/bin/python -u scripts/yolo_gpu.py --output "$CPG_RESULTS/measured"
apt-get update -qq
apt-get install -y -qq nsight-systems-2025.3.2
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi -o "$CPG_RESULTS/trace" .venv-gpu/bin/python -u scripts/yolo_gpu.py --engine "$CPG_RESULTS/measured/yolo.engine" --trace --output "$CPG_RESULTS/traced"
nsys export --type sqlite -o "$CPG_RESULTS/trace.sqlite" "$CPG_RESULTS/trace.nsys-rep"
for candidate in cpg torch cvcuda; do
  .venv-gpu/bin/python scripts/summarize_trace.py "$CPG_RESULTS/trace.sqlite" --range "yolo_${candidate}_complete" --expected-ranges 2 --output "$CPG_RESULTS/trace-${candidate}.json"
done
