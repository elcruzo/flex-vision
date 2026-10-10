# CUDA Preprocess Graph

**Camera frame → inference-ready GPU tensor.**

Measured experimental path on an RTX 4090:

```text
Resident 1920 × 1080 BGR8 frame
    → letterbox + RGB + scaling + FP16 NCHW 640 × 640
    → TensorRT YOLOv8n
    → TorchVision CUDA NMS
```

| Implementation | Preprocess p50 / p99 | Complete host p50 / p99 |
| --- | --- | --- |
| CPG | 0.084 / 0.106 ms | 0.869 / 1.079 ms |
| PyTorch | 0.100 / 0.162 ms | 0.889 / 1.070 ms |
| Two-pass CV-CUDA | 0.113 / 0.175 ms | 0.907 / 1.117 ms |

This table uses the person fixture from [experiment 021](docs/experiments/021-detector-repeatability.md), with 10,000 samples per candidate.
The preprocessing gain repeated across rentals. Complete p99 did not improve against PyTorch in this table.
The trace recorded no frame-sized host transfer in completed ranges, but small metadata transfers remained.
These are resident synthetic frames, not live-camera or ROS latency.

**Status: experimental fixed CUDA detector and inspection paths. Full runtime, platform, and release acceptance remain pending.**

The [inspection comparison](docs/experiments/014-inspection-comparison.md) measured 42% lower 4K preprocessing latency through a real classifier.
Small-image preprocessing was slower. See the [experimental inspection contract](docs/INSPECTION_RUNTIME.md).
The [30-minute inspection soak](docs/experiments/019-inspection-soak.md) passed on an L4.
The [short detector load matrix](docs/experiments/021-detector-repeatability.md) checked 187,279 completed frames and recorded zero declared post-drain allocator growth in its final run.
Earlier vendor pool-growth failures remain in that report. Detector throughput and tail results varied across blocks.
Read the [CUDA backend contract](docs/CUDA_BACKEND.md) for supported inputs, synchronous execution, and limitations.
The [30-minute detector soak](docs/experiments/022-detector-soak.md) passed on RTX 4090: 432,000 correct completed frames, zero drops, and zero declared allocator growth.
The separate synchronous caller-owned FP16 output scenario passed six TensorRT fixtures. The matched reuse comparison removed pool requests but increased preprocessing host p50 about 5.2%. Reuse remains optional.
Jetson, ROS transport, broader vendor comparisons, and full runtime acceptance remain open.

CUDA Preprocess Graph (CPG) aims to optimize the complete preprocessing pipeline.
It should keep GPU input on the GPU, reuse memory, and combine compatible operations.
The first users are perception engineers who work with Jetson, ROS 2, and NVIDIA GPUs.

## Project names

This repository is `elcruzo/flex-vision`. The earlier working name was FluxVision Runtime.
CUDA Preprocess Graph is the current product direction.
The proposed Python distribution is `cuda-preprocess-graph`, with the import name `cpg`.
The proposed ROS package is `cuda_preprocess`. These names do not imply published packages.

## Intended API

The graph-building calls below exist. Execution through `pipeline(frame)` uses the initial synchronous CUDA backend.

```python
import cpg

pipeline = (
    cpg.Pipeline(input_encoding="bgr8")
    .letterbox(640, 640, value=114)
    .normalize(mean=[0, 0, 0], std=[1, 1, 1], scale=1 / 255)
    .to(dtype="float16", layout="NCHW")
)

tensor = pipeline(frame)
```

The pipeline records operations before execution.
The current planner selects a fixed detector strategy. General operator planning and autotuning remain planned.
The runtime executes the fixed fused path on a CUDA stream.

## Required behavior

- GPU input must not require a CPU copy of the pixel payload.
- CuPy and PyTorch CUDA tensors must work through explicit memory and stream contracts.
- DLPack and the CUDA Array Interface must preserve buffer ownership and synchronization.
- Output must be usable by inference without an intermediate host copy.
- Performance reports must contain measured results and disclose unsupported metrics.
- CPU input must use an explicit upload path with measured transfer costs.

The initial operator scope includes resize, letterbox, crop, RGB/BGR conversion, normalization, type conversion, and layout conversion.
It also includes small custom stencils, Gaussian, sharpen, Sobel, clamp, and simple arithmetic.
An operator enters the release only after correctness and pipeline tests pass.

## Local development

The first slice provides Python/YAML graph construction, plan inspection, independent references, and a pretrained detector validation runner.
Read [the local development guide](docs/LOCAL_DEVELOPMENT.md) for setup, exact semantics, and reproduction commands.
The detector runner checks synthetic inputs and pinned photographs with expected objects.
A [complete inspection reference](docs/INSPECTION_REFERENCE.md) also checks Gaussian, sharpen, crop, resize, and normalization through a pretrained classifier.
Reference execution is explicit. It does not silently replace the planned CUDA path.

## Development strategy

Start with a detector input pipeline and an inspection pipeline.
Measure straightforward GPU baselines before expanding the operator set.
Test whether fusion, memory reuse, and GPU transport improve the complete workload.
Use vendor implementations when they produce a better plan.

The [requirements map](docs/REQUIREMENTS.md) preserves the complete product scope and acceptance checks.
The [delivery plan](docs/PLAN.md) defines goals and exit criteria.
The [design notes](docs/DESIGN.md) record proposed contracts and open decisions.
The [benchmark protocol](docs/BENCHMARKS.md) defines the evidence needed for performance claims.
The [hardware plan](docs/HARDWARE.md) explains cloud GPU and Jetson setup.
The [end-to-end test plan](docs/TESTING.md) defines the real inference feedback loop.
The [options register](docs/OPTIONS.md) retains alternative approaches and later possibilities.
The [decision log](docs/DECISIONS.md) separates accepted direction from provisional choices.

## Origin and scope

The project grows from [cuda-conv](https://github.com/elcruzo/cuda-conv).
That project is a source for stencil implementations and benchmark techniques.
This repository does not yet contain its code.
Before reuse, we will check the source revision, license, numerical behavior, and relevant tests.

CPG focuses on the boundary between camera data and model input.
It does not aim to replace OpenCV or provide a general tensor framework.
An experimental TensorRT consumer exists. ROS transport, production inference integration, and package distribution remain pending.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before implementation.
Use small commits with one purpose and evidence for the change.
Update the plan when experiments invalidate a proposed approach.

The intended project is open source. The project license remains an explicit decision before code import or distribution.
The bundled writing skill retains its own MIT license.
