# TensorRT detector experiment

`scripts/tensorrt_detector.py` tests a fixed SSDLite consumer for CPG's FP32 output.
It is an experiment, not a general TensorRT integration API.

## Inference boundary

```text
GPU BGR8 frame
  → CPG fused letterbox, RGB conversion, scale, and NCHW output
  → TensorRT: trained normalization, backbone, dense detection heads
  → TorchVision CUDA: box decode and non-maximum suppression
  → detections
```

The TensorRT context binds the CuPy input's existing device pointer with `set_tensor_address`.
No intermediate input buffer or host tensor is constructed for this handoff.
Output buffers are Torch CUDA tensors. The consumer retains them through stream completion.
The current consumer synchronizes before returning. It is not an asynchronous production adapter.

TensorRT runs the actual trained network. The experiment does not substitute an identity engine or random weights.
The detector retains its trained normalization. CPG's external scale remains 1/255, with mean zero and standard deviation one.
The complete model and the TensorRT path receive equivalent preprocessed images.

The first engine uses fixed shape `1×3×320×320`, FP32 I/O, strong typing, a 1 GiB workspace limit, and disabled TF32.
Decode and NMS remain in TorchVision on CUDA. This is a hybrid detector, not a complete TensorRT-only detector.
FP16 TensorRT I/O, dynamic shapes, batching, and YOLO remain separate acceptance tasks.

## Pinned setup

Use the recorded CUDA 13 image and framework pair from [experiment 004](experiments/004-cuda-host.md).
Install the experiment dependencies in its separate environment:

```bash
.venv-gpu/bin/python -m pip install numpy==2.2.6 cupy-cuda13x==14.2.0 onnx==1.19.1 tensorrt-cu13==10.13.3.9.post1 'cuda-toolkit[cudart]==13.0.0'
.venv-gpu/bin/python -m pip check
```

The original `tensorrt-cu13==10.13.3.9` package depends on the deprecated `nvidia-cuda-runtime-cu13` package and failed installation.
The `.post1` package uses `cuda-toolkit[cudart]` instead.
Toolkit package version 13.0.0 selects runtime 13.0.48, matching the template's PyTorch dependency.
This package pin selects runtime libraries. It does not replace the host driver or the installed CUDA compiler.

Download the pinned SSDLite weights using [the existing instructions](LOCAL_DEVELOPMENT.md).
The runner validates their full SHA256 before export.
No training, LLM service, or paid model API is needed.

TensorRT 10.13.3 documents CUDA 13.0 support. Recheck the compatibility matrix before changing versions.
[NVIDIA release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html)

## Run the full experiment

```bash
export PATH=/usr/local/cuda/bin:$PATH
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=benchmark-results/tensorrt-trace .venv-gpu/bin/python scripts/tensorrt_detector.py --output benchmark-results/tensorrt-detector
nsys export --type sqlite --output benchmark-results/tensorrt-trace.sqlite benchmark-results/tensorrt-trace.nsys-rep
python scripts/check_cuda_trace.py benchmark-results/tensorrt-trace.sqlite --expected-ranges 4 --stage preprocessing --output benchmark-results/cpg-trace-check.json
python scripts/check_cuda_trace.py benchmark-results/tensorrt-trace.sqlite --expected-ranges 4 --stage tensorrt --output benchmark-results/trt-trace-check.json
```

Use fresh output paths. The runner exports ONNX, checks the graph, and builds the engine on the target GPU.
The legacy PyTorch exporter is explicit (`dynamo=False`) with fixed shapes and ONNX opset 17.
Its fixed-shape tracing warnings do not imply dynamic-shape support.
Profiler capture starts after engine construction, model upload, and inference warmup.

The runner compares dense predictions, high-confidence labels, scores, boxes, and expected-object overlap on four pinned photo cases.
Tolerances are fixed before execution: dense absolute `0.01` and relative `1e-4`, scores `0.001`, boxes `0.1`.
Validation downloads occur after the measured handoff range.
The `cpg_to_tensorrt` trace check covers preprocessing through dense TensorRT output, including stream completion.
It excludes TorchVision decode/NMS, oracle execution, and validation transfers.
The narrower `cpg_preprocess` range must still contain exactly one kernel.

Save the engine, ONNX model, hashes, reports, package inventory, and trace before terminating compute.
Do not assume an engine built on one GPU or TensorRT version is portable to another.
The experiment does not yet provide engine cache compatibility checks or an engine-loading API.

## Limits

A successful run establishes one fixed detector handoff and its numerical output.
It does not establish ROS transport, camera capture, a latency advantage, tail latency, or sustained multi-camera throughput.
The CPU still schedules work. Zero pixel transfers at this boundary do not mean zero CPU use.
The full requirements and benchmark success gate remain unchanged.
