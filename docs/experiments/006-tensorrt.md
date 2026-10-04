# First TensorRT detector handoff

Result: four photograph cases passed the fixed FP32 TensorRT consumer on an NVIDIA L4.
CPG passed its existing 48 GPU numerical, stride, and output-retention cases on this device.
This establishes correctness for one hybrid detector. It does not establish a latency advantage or full G1 acceptance.

The traced detector ran at clean revision `a9412aa9da264a1100ed4ebfec29a451e2222943`.
Read [the experiment procedure](../TENSORRT_EXPERIMENT.md) for setup and reproduction commands.

## Actual inference and correctness

CPG converts GPU BGR8 input into a fixed FP32 RGB tensor with shape `1×3×320×320`.
TensorRT binds that CuPy tensor's device pointer directly.
The engine executes SSDLite's trained normalization, backbone, and dense detection heads.
TorchVision CUDA performs box decoding and non-maximum suppression after the engine completes.
This is a real pretrained detector, with a hybrid inference path.

All four CPG tensors matched the independent NumPy reference exactly in this run.
Dense predictions, high-confidence labels, scores, boxes, and semantic smoke expectations passed the predefined tolerances.

| Case | Maximum dense absolute error | Expected-object source-space IoU |
| --- | --- | --- |
| astronaut | 0.000026942 | 0.930434 |
| astronaut-padded | 0.000030518 | 0.863174 |
| chelsea | 0.000025272 | 0.944010 |
| chelsea-padded | 0.000030518 | 0.916463 |

Dense tolerances were absolute 0.01 and relative 0.0001. Score tolerance was 0.001 and box tolerance was 0.1 pixels.
Labels above confidence 0.5 had to match. The expected object needed an IoU of at least 0.5.
These four smoke cases do not establish dataset accuracy.

## Trace evidence

Each of four synchronous `cpg_preprocess` ranges contained one kernel and zero memory-copy events.
Each enclosing `cpg_to_tensorrt` range contained 167 kernels and zero memory-copy events or transferred bytes.
The enclosing range ends after the dense TensorRT network completes.
It excludes TorchVision decoding/NMS, input uploads, independent oracle execution, and validation downloads.
These observations do not establish zero-copy camera capture or ROS transport.

The trace checker passed on the GPU host and again on the Mac after artifact transfer.
A negative control rejected the earlier preprocessing-only trace because it lacked TensorRT ranges.
Nsight capture began after engine construction and warmup.

TensorRT warned that default-stream execution can add synchronization.
The adapter also synchronizes explicitly to preserve readiness and buffer lifetimes.
The next comparison must use an explicit stream and measure host cost as well as GPU events.
Engine construction took 51.24 seconds. This diagnostic value is not inference latency.

## Environment and artifacts

- NVIDIA L4, compute capability 8.9, driver 595.91.07.
- CUDA compiler 13.0.88, Python 3.12.3, Nsight Systems 2025.3.2.
- PyTorch 2.9.1+cu130, TorchVision 0.24.1+cu130, CuPy 14.2.0.
- TensorRT 10.13.3.9.post1, ONNX 1.19.1, CUDA toolkit Python package 13.0.0.
- Image: `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`.

The first TensorRT package installation failed on a deprecated runtime dependency.
The documented `.post1` package and matching runtime pin resolved it. `pip check` passed.
The hardware preflight's pending fields describe its limited arithmetic scope. The separate detector and trace reports provide inference evidence.

All 16 exported files matched their remote SHA256 checksums before compute termination.
Local artifact directory: `benchmark-results/iteration-006-tensorrt/`.
The bundle includes engine, ONNX, numerical outputs, reports, package inventory, and Nsight files.
Large artifacts remain outside Git.

| Artifact | SHA256 |
| --- | --- |
| Detector report | ddff47656b1f530c04cbd4bc1866491fe5e8df9a53d9498abcbb992893608098 |
| TensorRT engine | 8f8039d712b33e90c1447afb788ff43e8829eafafae31204a7716bce6d65d551 |
| ONNX model | 59ef97dd97cc0a4f17af0e484d4aeccbb4ea1e431c404c2679db04d4ec413863 |
| Nsight report | 6766bcadb12bec10fd2c24e573ebed7e73196bea058b382fa195822869b962ca |
| Trace SQLite | d08015801f9bc448ac96721b4ba7f7b949e5b5655f40502397d9fde51ed3431b |

## Cost and cleanup

Pod `dsaslqanvq6ka9` ran in EU-RO-1 at $0.49/hour with 50 GB temporary disk and no persistent volume.
It started at 17:18:17 UTC and was terminated at approximately 17:35:15 UTC on 2026-10-04.
Compute cost from elapsed time is approximately $0.139. Reserve $0.15 for this session, including temporary disk, until billing posts.
The billing read returned no posted records for this Pod at cleanup. This does not mean the session was free.
Earlier posted charges total $0.15281239314936102, giving approximately $0.303 including this session's reserve against the $15 authorization.
The account read after termination returned no Pods. The local fallback termination timer was cancelled.

## Next experiment

Use an explicit stream for the same TensorRT consumer and rerun correctness and trace checks.
Compare fused CPG preprocessing against equivalent unfused CUDA preprocessing through that same engine.
Save raw warm samples for preprocessing, inference, and complete pipeline latency, with p50, p95, and p99.
Separate compilation, engine construction, allocation, synchronization, and verification from measured warm execution.
Keep the two-workload success gate unchanged. Classifier, YOLO, FP16 TensorRT, asynchronous ownership, batching, Jetson, and ROS remain pending.
