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

[Experiment 006](experiments/006-tensorrt.md) records the successful L4 run, scoped trace evidence, artifacts, and limitations.

## Matched serial latency experiment

The detector runner now uses an explicit shared Torch/CuPy stream for warmup and inference.
The benchmark uses that same stream policy for both candidates.
[NVIDIA's inference procedure](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html) describes pointer binding, stream submission, and completion requirements.

First run the detector validation on the current GPU. Then use its engine and report:

```bash
.venv-gpu/bin/python scripts/tensorrt_benchmark.py --validated-run benchmark-results/tensorrt-detector --output benchmark-results/latency
```

The default collects 10,000 samples per candidate across ten alternating-order repeats, after 100 warmup calls per candidate per repeat.
It saves every sample and per-repeat p50/p95/p99 with NumPy's linear percentile method.
A separate diagnostic run can use `--trace --samples 2 --repeats 1 --warmup 1` under Nsight's CUDA-profiler capture mode.
Instrumented timings must not become headline benchmark numbers.

Input is a pinned photograph enlarged by exact pixel repetition, then centered in a 1920×1080 frame.
The output remains the existing 320×320 FP32 SSDLite contract. This does not replace the required 640×640 FP16 YOLO workload.
The runner checks preprocessing tensors and actual detections against the independent reference before measuring either candidate.

The unfused baseline uses PyTorch flip, permutation, float conversion, bilinear resize, padding, normalization, and contiguous output.
Mean and standard deviation remain cached on the device. Both candidates allocate fresh output arrays through warmed framework pools.
Both complete preprocessing synchronously before the same TensorRT consumer. Both include the same CUDA decoder and NMS.

CUDA events delimit preprocessing, dense inference, and decoding/NMS on the shared stream.
These intervals can include host dispatch gaps and synchronization effects. They are not sums of kernel execution durations.
Host timing starts before preprocessing and ends after decode/NMS completion.
Uploads, file decoding, engine construction, correctness downloads, warmup, and CSV writes remain outside the sampled interval.
This serial replay does not measure live-camera latency, queue behavior, or multi-camera throughput.
The allocation policy and synchronous implementation remain limitations for performance interpretation.

Recompute the completed run on a CPU host:

```bash
python scripts/summarize_latency.py benchmark-results/latency --output benchmark-results/latency-summary.json
```

This command verifies sample hashes, unique identities, counts, balance, and finite positive latencies.
It rejects failed or instrumented runs. It reports pooled percentiles and paired per-repeat reductions.
Observed repeat ranges are not confidence intervals. Keep all samples, including outliers.

For a separate short Nsight capture, count work inside each complete hybrid detector range:

```bash
python scripts/summarize_trace.py benchmark-results/latency-trace.sqlite --range cpg_hybrid_pipeline --expected-ranges 2 --output benchmark-results/cpg-hybrid-trace.json
python scripts/summarize_trace.py benchmark-results/latency-trace.sqlite --range torch_unfused_hybrid_pipeline --expected-ranges 2 --output benchmark-results/baseline-hybrid-trace.json
```

The trace summary reports copies by CUPTI kind. It does not assume that decoder metadata transfers contain image payloads.
The strict zero-copy check for preprocessing through dense TensorRT output remains a separate check.

## Inference attribution controls

Use short diagnostic runs before changing the runtime to explain downstream timing differences.
These runs do not replace the 10,000-sample performance comparison.

- `--control fixed-input`: execute each preprocessing path, but feed TensorRT the same retained device tensor and pointer.
- `--control no-preprocess`: both candidate labels return that same retained tensor. Neither label executes preprocessing during timing.
- `--reuse-outputs`: reuse the TensorRT dense output buffers. The next inference overwrites their contents.

The real preprocessing paths still pass numerical and detection validation before any control runs.
Control names and buffer policies appear in the report and CPU summary.
Only the serial experiment uses output reuse. The library's fresh-output ownership contract remains unchanged.
Every decode must complete before the next inference overwrites the dense outputs.

Run the unchanged baseline and each control with the same engine, sample count, warmup, and repeat count:

```bash
python scripts/tensorrt_benchmark.py --validated-run benchmark-results/tensorrt-detector --output benchmark-results/control-fixed --control fixed-input --samples 300 --repeats 4 --warmup 50
python scripts/tensorrt_benchmark.py --validated-run benchmark-results/tensorrt-detector --output benchmark-results/control-empty --control no-preprocess --samples 300 --repeats 4 --warmup 50
python scripts/tensorrt_benchmark.py --validated-run benchmark-results/tensorrt-detector --output benchmark-results/control-fixed-reuse --control fixed-input --reuse-outputs --samples 300 --repeats 4 --warmup 50
```

Also repeat `--control no-preprocess` with `--reuse-outputs` to separate label/order effects from preprocessing effects.
A difference between the no-preprocessing labels is measurement variation or harness behavior, not a CPG improvement.
A fixed-pointer difference cannot be explained by different inference input values or addresses alone.
Output reuse removes dense-output allocation, but does not remove allocations inside the decoder.
Keep unexplained effects explicit rather than selecting only the fastest control.

An additional memory-state control is available:

```bash
python scripts/tensorrt_benchmark.py --validated-run benchmark-results/tensorrt-detector --output benchmark-results/control-conditioned --control fixed-input --reuse-outputs --condition-memory-mib 256 --samples 300 --repeats 4 --warmup 50
```

Both candidates write the same scratch buffer immediately before dense inference, then synchronize.
The kernel writes varied integer values to avoid an all-zero compression case.
This is common memory conditioning, not a guaranteed cache flush or a production optimization.
Its cost is included in the preprocessing interval, so that interval cannot represent ordinary preprocessing performance.
Compare the downstream network and decoder intervals with the fixed-input, reused-output control.
A reduced difference supports a preceding-memory-state effect. It does not establish a specific cache mechanism without additional counters.
The scratch allocation is bounded to 1024 MiB and is available only with the fixed-input control.
