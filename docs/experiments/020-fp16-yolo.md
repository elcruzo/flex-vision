# Experiment 020 — FP16 YOLO through TensorRT

Status: revision 4 passed numerical acceptance and the matched timing comparison on an RTX 4090.
The trace capture exists, but SQLite export failed after the profiler terminated its target. Transfer verification remains pending.

## Contract and controls

Measured source `cdc4dc28f59a87e8cc4176f8787500fdf0bc363f` runs the [fixed YOLO protocol](../YOLO_VALIDATION.md).
The pipeline converts resident BGR8 HWC input to centered 640×640 FP16 RGB NCHW with scale 1/255.
The consumer uses a strongly typed TensorRT YOLOv8n engine and shared TorchVision CUDA NMS.
The environment pins CUDA 13, PyTorch 2.9.1, CV-CUDA 0.18.0, and TensorRT 10.13.3.9.post1.
The vendor baseline uses two passes: resize/convert/reformat, then planar padding.

All six original, padded, and derived 1080p person/cat cases passed all five input paths.
CPG tensors were bitwise equal to the independent NumPy FP16 oracle. Both library baselines differed by at most one FP16 step on smaller fixtures.
Both 1080p fixtures produced identical tensors for all candidates. CPG's matched detection IoU was 1.0 on every case.
An independent NumPy check repeated all 18 main candidate comparisons, including dense outputs and CPU NMS.
Its minimum baseline detection IoU was 0.99927. These smoke fixtures do not establish dataset accuracy.

The earlier model-preparation and acceptance failures remain in the protocol and retained data.
Revision 4 does not retroactively pass those runs. It keeps exact CPG checks and explicit approximation limits for library baselines.

## Matched timing

Each candidate/fixture has 10,000 samples, split into ten blocks with rotated candidate order after 100 warmups.
All 60,000 raw samples passed independent coverage, ordering, finite timing, and component-sum checks.
Times include synchronous preprocessing with fresh output allocation. The complete host boundary includes TensorRT and CUDA NMS.

| Fixture | Candidate | Preprocess p50 / p95 / p99, ms | Complete host p50 / p95 / p99, ms |
| --- | --- | --- | --- |
| Person 1080p | CPG | 0.1078 / 0.1178 / 0.1454 | 0.8958 / 0.9471 / 1.1236 |
| Person 1080p | PyTorch | 0.0993 / 0.1136 / 0.1679 | 0.8873 / 0.9377 / 1.1046 |
| Person 1080p | CV-CUDA | 0.1116 / 0.1260 / 0.1577 | 0.9030 / 0.9484 / 1.0507 |
| Cat 1080p | CPG | 0.1078 / 0.1208 / 0.1495 | 0.8965 / 0.9703 / 1.2137 |
| Cat 1080p | PyTorch | 0.0995 / 0.1165 / 0.1617 | 0.8879 / 0.9523 / 1.1700 |
| Cat 1080p | CV-CUDA | 0.1118 / 0.1352 / 0.1946 | 0.9037 / 0.9882 / 1.2976 |

CPG preprocessing p50 was 8.4–8.5% slower than PyTorch and 3.4–3.6% faster than CV-CUDA.
The cat preprocessing p99 was 23.2% lower than CV-CUDA, but person complete p99 was 6.9% higher.
This isolated tail result does not establish a repeatable second-workload advantage or complete the product success gate.
Complete median changes were about one percent. Preserve this mixed result rather than claiming a broad speedup.
Serial throughput here is reciprocal mean service time, not sustained multi-camera capacity.

## Trace and cleanup

Nsight captured all requested ranges, but its default capture shutdown sent SIGTERM at `cudaProfilerStop`.
The script therefore exited 143 before SQLite export and final trace-report persistence.
No verified transfer or launch count is claimed yet. The next capture uses `--capture-range-end=stop --kill=none`.
[NVIDIA's guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) defines this behavior.

All 61 remote artifacts matched SHA-256 hashes before GPU termination at 05:20:52 UTC on October 9.
Controller and temporary credential removal passed at 05:21:03 UTC. The lease stayed within its independent 60-minute deadline.
The [retained evidence](data/020/completed/results.json) includes raw compressed samples, environment, verifier results, and the remote artifact manifest.
Model binaries, dense arrays, and the Nsight capture remain in the ignored local result bundle.

## Next hypothesis and limits

CPG rebuilds graph inspection dictionaries and scalar launch arguments for every frame.
Cache only immutable host metadata by graph and input shape, with bounded capacity. Keep strides, buffers, and stream owners outside the cache.
Rerun identical numerical and timing checks. Do not change interpolation or precision to hide the initial result.

This experiment does not validate ROS transport, live cameras, Jetson, asynchronous dispatch, reusable output workspace, batching, or detector soak behavior.
GPU bandwidth, power, CPU utilization, peak temporary memory, and allocator stability were not measured by this run.
