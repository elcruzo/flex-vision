# CUDA Preprocess Graph

**Camera frame → inference-ready GPU tensor.**

First benchmark target:

```text
1920 × 1080 RGB8 camera frame on the GPU
    → letterbox to 640 × 640
    → normalize
    → FP16 NCHW tensor
    → TensorRT inference
```

**Status: initial fused CUDA detector preprocessing, plus a measured CUDA inspection reference. CPG's two-workload performance gate and full-platform acceptance remain pending.**

Read the [CUDA backend contract](docs/CUDA_BACKEND.md) for supported inputs, synchronous execution, and limitations.

The [fused CUDA experiment](docs/experiments/005-fused-cuda.md) passed real detector checks.
The [inspection CUDA baseline](docs/experiments/012-inspection-cuda.md) passed real MobileNetV3 inference and a zero-copy Nsight range; its measurements are reference evidence, not CPG gains.
The [matched inspection experiment](docs/experiments/014-inspection-comparison.md) measured 42% lower 4K preprocessing latency with crop-aware execution through the same classifier.
This is an experimental PyTorch planner candidate. Small-image preprocessing was slower, and public CPG integration remains pending.
Its trace shows one kernel and no memory-copy events within each preprocessing call.
The [TensorRT experiment](docs/experiments/006-tensorrt.md) also passed on an L4, with no copies through the dense network boundary.
The [first matched latency experiment](docs/experiments/007-latency.md) records 10,000 samples per candidate through a hybrid TensorRT detector.
It shows a preprocessing gain on one L4 workload. [Attribution controls](docs/experiments/008-controls.md) support a preceding-memory-state effect in downstream timings.
Vendor comparisons and the two-workload success gate remain pending.

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
    cpg.Pipeline()
    .letterbox(640, 640, value=114)
    .normalize(mean=[0, 0, 0], std=[1, 1, 1], scale=1 / 255)
    .to(dtype="float16", layout="NCHW")
)

tensor = pipeline(frame)
```

The pipeline records operations before execution.
A planner chooses implementations, compatible fusion groups, and temporary buffers.
The runtime then executes that plan on a CUDA stream.

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
