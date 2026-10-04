# Benchmark protocol

Status: [one matched L4 replay](experiments/007-latency.md) has raw samples and scoped trace evidence.
The required detector, inspection, and multi-camera benchmark matrix remains incomplete.
[TESTING.md](TESTING.md) defines the runnable harness requirements and the change–measure–rerun loop.
[HARDWARE.md](HARDWARE.md) defines environment setup and which claims each target can validate.

## Questions

Does CPG improve complete preprocessing compared with a straightforward GPU implementation?
Which benefit comes from fusion, buffer reuse, or transport?
Does that benefit remain under concurrent camera load?

## Workloads

| Workload | Input and operations | Required output |
| --- | --- | --- |
| Detector | 1080p RGB8 → letterbox 640×640 → normalize | FP16 NCHW detector input |
| Inspection | 4K image → Gaussian → sharpen → crop → resize → normalize | Classifier input with recorded shape and dtype |
| Robot | Multiple 1080p streams → rectify → custom stencil → resize → normalize | Inference input with per-camera timestamps |

Record exact crop coordinates, filter coefficients, calibration, channel encoding, and model requirements in each benchmark configuration.
Rectification can use a vendor implementation. Disclose its cost and transport boundaries.
Run single-image tests first, then batches and sustained camera streams.

## Baselines

Start with straightforward CuPy and PyTorch implementations.
Add CV-CUDA, VPI, OpenCV CUDA, Kornia/Kornia-rs, and Isaac ROS Image Proc where the workload and platform permit them.
Record library versions, supported operations, and reasons for unavailable comparisons.
Do not require every library to implement an identical public API.
Require equivalent image semantics and the same final tensor contract.

Use vendor-recommended execution settings where practical.
Publish the baseline code and its allocation policy.
Include both idiomatic and reused-buffer baselines when allocation reuse changes the result materially.
A CPU reference checks correctness. It does not establish a competitive GPU performance claim.

## Measurement procedure

1. Record the Git revision and all dependencies.
2. Record GPU, driver, toolkit, OS, clocks, power mode, and thermal state.
3. Validate each candidate against the independent numerical reference.
4. Separate compilation and cold-start measurements from steady-state runs.
5. Warm each candidate until timings and memory use stabilize.
6. Collect raw per-frame samples with explicit synchronization boundaries.
7. Repeat runs and alternate candidate order to reduce thermal bias.
8. Save failures, dropped frames, and excluded samples with reasons.
9. Publish the exact commands, configurations, and raw result artifacts.

Proposed default: at least 10,000 measured frames per candidate across multiple runs for tail-latency analysis.
Adjust the sample count to the workload duration and report the change.
Report per-run results and uncertainty. Do not report only the best run.
Keep clocks and power policy equal across compared candidates.
Record whether inputs are synthetic, recorded, or live camera frames.
Do not include file decoding in preprocessing latency unless the workload explicitly requires it.

## Timing boundaries

Use CUDA events for GPU execution on the measured stream.
Use a monotonic host clock for application latency and queue delay.
Record event dependencies for work that crosses streams.
Do not measure asynchronous enqueue time as completed GPU latency.

Report preprocessing separately from inference.
Measure camera-to-model latency from a documented camera timestamp to a documented inference completion boundary.
Check timestamp clock domains before subtracting them.
Include capture, upload, queues, preprocessing, inference, and synchronization where the endpoint requires them.

Measure sustained throughput with bounded queues and a fixed offered load.
Report per-camera latency, fairness, and dropped frames.
Do not infer multi-camera throughput by dividing one by a single-frame latency.

## Required report fields

| Field | Measurement or reporting rule |
| --- | --- |
| p50 / p95 / p99 | Raw latency samples, count, percentile method, timing boundary |
| Images per second | Completed images divided by elapsed wall time |
| Temporary GPU memory | Live workspace peak, separately from input/output and allocator reserve |
| Allocations | Cold and steady-state counts with allocator definition |
| CUDA launches | Trace-derived count, including vendor library work |
| Host/device bytes | Trace-derived transfers, including hidden staging |
| GPU bandwidth | Hardware counters or a labeled byte-traffic estimate |
| CPU utilization | Sampling interval, process scope, core-count convention |
| GPU utilization | Tool, sampling interval, and scope |
| Jetson power | Measurement source, power mode, interval, temperature |
| Camera-to-model latency | Explicit start/end events and timestamp domains |
| Dropped frames | Offered, accepted, completed, and discarded counts |

Mark an unavailable metric as unavailable with a reason.
Estimated traffic is not measured physical memory bandwidth.
Keep trace collection separate from headline timing runs because instrumentation can affect latency.
For supported zero-host-copy claims, include an Nsight Systems trace and instructions to inspect it.

## Experiments that explain the result

Compare the baseline, reused-buffer path, fused path, and selected vendor plan independently.
For ROS, compare CPU transport and GPU transport with the same kernels.
For stencils, compare generic, specialized, tiled, and separable candidates where valid.
Test Orin, Thor, and a desktop NVIDIA GPU when hardware becomes available.
Do not generalize one GPU result to an untested platform.

## Publication gate

Publish a performance claim only when the raw evidence reproduces it.
Reference the exact configuration, code revision, environment, and numerical tolerance beside the claim.
Use the success criteria in [PLAN.md](PLAN.md) to decide whether to expand the project.
Keep unsuccessful comparisons in the report.
Store large traces as release or benchmark artifacts rather than normal Git history.
