# Product requirements and acceptance map

Status: planned. No requirement below represents implemented runtime support.
This document preserves the CUDA Preprocess Graph brief at release-level detail.
[PLAN.md](PLAN.md) sequences the work. [DESIGN.md](DESIGN.md) records technical choices.

A requirement states the result the product must deliver.
A candidate implementation states one way to achieve that result.
Changing an implementation does not remove its requirement.
Record a proposed scope reduction explicitly before treating the release as complete.

## Customer and product boundary

The first customer is a perception engineer with camera streams, GPU models, and unexplained preprocessing overhead.
Their systems include humanoids, industrial arms, warehouse robots, mobile robots, drones, and factory inspection equipment.
Typical inputs come from USB, CSI, or GigE cameras.
Typical tools include ROS 2, Jetson, CUDA, TensorRT, PyTorch, OpenCV, and Isaac ROS.

A representative conversion is uint8 HWC 1920×1080 BGR into FP16 NCHW 640×640 RGB.
Existing pipelines can contain CPU stages, repeated allocations, layout conversions, and unnecessary transfers.
CPG must replace that integration code with a declared pipeline and an optimized execution plan.
The engineer should not need to select shared-memory tiles or kernel block sizes.

Custom preprocessing that vendor pipelines do not already cover is a primary use case.
The product should compose with vendor rectification and other established stages.
It need not replace those stages to be useful.

The output can feed detection, segmentation, depth, hand/object interaction, or VLM image encoders.
Inspection uses include defect classification, surface segmentation, OCR, barcode detection, and pose estimation.
These are downstream use cases, not separate model implementations promised by CPG.

## Requirements map

All rows are pending. Milestones identify delivery responsibility, not proof of completion.

| ID | Required result | Milestone | Acceptance evidence |
| --- | --- | --- | --- |
| R01 | Reuse relevant cuda-conv work | G0, G3 | Pinned source audit, attribution, retained tests, stencil comparison |
| R02 | Construct pipelines without execution | G1 | Construction launches no kernels and allocates no frame workspace |
| R03 | Support Python and YAML descriptions | G1, G4 | Equivalent descriptions produce equivalent graphs and tensors |
| R04 | Preserve GPU input residency | G1, G5 | Payload transfer trace covers input through inference handoff |
| R05 | Support CuPy, PyTorch CUDA, CUDA Array Interface, DLPack | G1, G6 | Each interface passes ownership, device, stride, and stream tests |
| R06 | Produce TensorRT-compatible output without a host copy | G1, G5 | Real TensorRT consumer uses output with correct lifetime and ordering |
| R07 | Deliver all listed MVP operators | G1, G3, G6 | Operator matrix below passes independent reference tests |
| R08 | Specialize user stencils | G3 | All required sizes work from Python and configuration |
| R09 | Plan fusion and implementation selection | G3, G4 | Inspectable plan preserves semantics and measures whole-graph cost |
| R10 | Reuse memory and support asynchronous streams | G3, G6 | Bounded steady-state workspace and concurrent-stream tests |
| R11 | Tune valid strategies and reuse the cached plan | G4 | Measured records, cache-hit execution, invalidation tests |
| R12 | Profile original and optimized graphs | G4 | Stage/group timings, totals, allocations, launches, and copies |
| R13 | Support CUDA-backed rosidl::Buffer in ROS 2 Lyrical | G0, G5 | Verified API contract and GPU transport experiment |
| R14 | Ship ROS Image and tensor paths with CPU-input support | G5 | GPU and CPU-input integration tests plus ordinary tooling instructions |
| R15 | Validate priority platforms | G0, G6 | Per-platform compatibility and test records |
| R16 | Reproduce three realistic benchmark workloads | G2, G5, G6 | Detector, inspection, and multi-camera robot reports |
| R17 | Compare credible GPU alternatives | G2, G3 | Baseline applicability matrix with results or explicit exclusions |
| R18 | Report all specified performance metrics | G2, G4, G5 | Measurement definitions and raw evidence, or unavailable reasons |
| R19 | Pass the two-pipeline success gate | G3 | Reproducible qualifying improvement on two realistic pipelines |
| R20 | Ship the Jetson flagship demo and trace | G5, G6 | Launch command, live metrics, trace, and reproduction instructions |
| R21 | Distribute through GitHub, PyPI, and ROS | G6 | Install checks, release documentation, ROS indexing work |
| R22 | Answer the research questions with controlled experiments | G2–G6 | Ablations and cross-platform results, including negative results |

## Python and configuration experience

These are target interfaces. They are not executable examples yet.

```python
import cpg

pipeline = (
    cpg.Pipeline()
    .gaussian(5, sigma=1.2)
    .custom_conv(kernel)
    .letterbox(640, 640, value=114)
    .normalize(mean, std, scale=1 / 255)
    .to(dtype="float16", layout="NCHW")
)
tensor = pipeline(frame)
```

The stencil interface must also accept literal coefficients:

```python
pipeline = cpg.Pipeline().conv2d([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
])
```

Resolve whether `custom_conv` and `conv2d` are aliases before stabilizing the API.
Both examples express the same required ability to supply a custom stencil.
The added explicit scale removes ambiguity about normalization of uint8 input.
Its exact schema remains a design choice.

Target `perception.yaml`:

```yaml
input:
  encoding: rgb8

pipeline:
  - gaussian:
      size: 5
      sigma: 1.2
  - convolution:
      kernel: kernels/sharpen.yaml
  - letterbox:
      width: 640
      height: 640
      value: 114
  - normalize:
      scale: 0.00392156862745098
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

output:
  dtype: float16
  layout: nchw
```

Include a matching kernel file and schema when implementing the example.
Test relative kernel paths, invalid coefficients, unknown fields, and Python/YAML equivalence.
Save model-specific preprocessing requirements with each model example.
Do not assume every detector uses this normalization.

## Complete MVP operator matrix

The proposed first release must include every row.
The detector milestone is an early subset, not the complete MVP.

| Operator | Delivery | Required checks |
| --- | --- | --- |
| Resize | G1 | Output dimensions, interpolation, coordinate and border rules |
| Letterbox | G1 | Aspect ratio, odd padding, pad value, reverse coordinate mapping |
| Crop | G3 | ROI bounds, strides, output dimensions |
| RGB/BGR conversion | G1 | Exact channel permutation and encoding |
| uint8 to FP16/FP32 | G1 | Scale policy, rounding, finite range, dtype |
| Normalize | G1 | Per-channel mean/std, explicit scale, zero-std rejection |
| HWC to CHW/NCHW | G1 | Element placement and explicit batch dimension |
| Custom convolution | G3 | 3×3, 5×5, 7×7, 9×9, asymmetric coefficients, border semantics |
| Gaussian | G3 | Size, sigma, reference pixels, separable/direct equivalence tolerance |
| Sharpen | G3 | Coefficients or strength, output range, clipping policy |
| Sobel | G3 | Gradient axes, signs, separate/magnitude output decision |
| Clamp | G3 | Endpoints, invalid bounds, supported dtype behavior |
| Simple arithmetic | G3 | Defined add/multiply scope, broadcasting, operation order |

The simple-arithmetic API needs an explicit supported-operation list before implementation.
Broader arithmetic can wait. The required basic capability cannot disappear behind that decision.
Batch RGB, supported dtypes, and concurrent streams require release tests across applicable operators.

Common constant stencils must have a compile-time specialization path with loop unrolling.
Check generated code or compiler evidence rather than inferring specialization from a function name.
Compare naive, shared-memory tiled, specialized, and separable strategies where each applies.
Tiling is a candidate strategy, not a requirement that every stencil use shared memory.

## Graph planning and backend selection

The required sequence is description → graph → planner → execution plan → CUDA execution.
The planner must expose operation groups, implementation names, workspace lifetimes, and stream dependencies.

Use sharpen → resize → normalize → FP16 → CHW as a fusion test case.
Compare its separate-stage path with valid fused plans.
Attempt to reduce passes without assuming that one kernel is always fastest.

Backend candidates include generated CUDA, CV-CUDA, NVIDIA VPI, specialized internal kernels, and CuPy operations.
The release need not implement every backend adapter.
It must support measured selection among its implemented valid strategies.
Document unsupported candidates and the reason for excluding them.

## Tuner and profiler contract

Target commands:

```bash
cpg tune perception.yaml
cpg profile perception.yaml
cpg benchmark examples/yolo.yaml
```

A tuning record must include:

- GPU architecture and device identity.
- Image dimensions, batch size, layout, dtype, and pixel format.
- Canonical pipeline and kernel coefficients or their stable hashes.
- Selected implementation and candidate parameters.
- Median, p95, and p99 latency with sample count and timing boundary.
- Temporary memory and kernel launch count.
- Software versions, numerical policy, and cache format version.

Production must load the matching cached plan.
Test cache hits, stale records, device changes, graph changes, and unavailable implementations.
Expose the fallback decision. Do not tune silently while processing production frames.

The profiler must show original stages and optimized groups with measured timings.
Include Gaussian, sharpen, and the letterbox/normalize/FP16/NCHW group in a representative report.
Show temporary allocations, launches, host copies, transferred bytes, and total latency for both plans.
Label instrumentation overhead and distinguish stage sums from independently measured total time.
Never copy illustrative latency values into a result report.

## ROS contract and Jetson demo

Ship the `cuda_preprocess` package and the `cuda_preprocess_node` component.
The brief uses `preprocess_node` in its CLI example.
Resolve the executable/component naming explicitly and test the documented invocation.

```bash
ros2 run cuda_preprocess preprocess_node --config perception.yaml
ros2 launch cuda_preprocess yolo_demo.launch.py
```

Input is `sensor_msgs/Image`.
Output is an Image when its representation fits, or a tensor message compatible with downstream inference.
Use ordinary ROS configuration and tooling. Do not require a proprietary graph format.

CUDA-backed `rosidl::Buffer` support in ROS 2 Lyrical is an explicit requirement to investigate and implement.
Its spelling, API, allocator, and transport behavior need verification against the installed ROS version.
An unsupported API is a reported blocker or scope decision, not evidence that a generic CUDA message satisfies this requirement.

CPU messages need a supported path with explicit upload costs.
CPU execution fallback is a separate optional design choice.
GPU-backed input must stay GPU-backed through preprocessing and inference handoff.
Test timestamps, ownership, queue limits, backpressure, and dropped-frame reporting.

The flagship chain is:

```text
CSI camera → ROS Image → cuda_preprocess_node
           → TensorRT detector → detections → perception stack
```

The Jetson demo must display measured camera dimensions and frame rate.
It must also display preprocessing time, inference time, total latency, CPU copy count, GPU memory, and dropped frames.
Define each metric's time window and scope.
Include an Nsight Systems trace with instructions to inspect payload residency and transfers.
A visual dashboard alone does not satisfy this deliverable.

## Platforms, benchmarks, and evidence

Priority targets are Jetson AGX Orin, Jetson AGX Thor, and x86-64 NVIDIA GPUs.
Requested integrations are CUDA 13, ROS 2 Lyrical, PyTorch, TensorRT, and CuPy.
Track each target explicitly. Do not infer compatibility across all combinations.
Older platforms are lower priority.

Required workload endpoints:

1. 1080p RGB → letterbox 640×640 → normalize → FP16 NCHW → TensorRT YOLO.
2. 4K camera → denoise → sharpen → crop ROI → resize → normalize → classifier.
3. Multiple 1080p cameras → rectify → custom convolution → resize → normalize → TensorRT.

The robot workload is a release deliverable even if two other pipelines satisfy the early success gate.
Select camera count, offered frame rate, queue policy, and duration explicitly.
Include a six-camera scenario when available hardware permits it.
Record the reason and actual load when hardware limits that scenario.

Compare CV-CUDA, VPI, Isaac ROS Image Proc, Kornia/Kornia-rs, OpenCV CUDA, and straightforward PyTorch/CuPy where applicable.
Record exclusions per workload and platform rather than omitting a baseline silently.

Report median, p95, p99, images per second, bandwidth, peak temporary memory, launches, CPU use, transfer bytes, GPU use, Jetson power, and camera-to-model latency.
[BENCHMARKS.md](BENCHMARKS.md) defines measurement boundaries and unavailable-metric reporting.

## Research questions

Working paper title: **Fused Zero-Copy Camera Preprocessing for GPU-Accelerated ROS 2 Perception**.
The paper follows evidence. It is not a prerequisite for the first release.

| Question | Required experiment |
| --- | --- |
| How much latency comes from preprocessing and data movement? | Attribute capture, queues, transfers, preprocessing, and inference separately |
| When does fusion beat vendor primitives? | Compare equivalent complete plans with fusion enabled and disabled |
| What happens to p99? | Measure sustained runs and contention, including per-camera tails |
| What does CUDA ROS transport contribute independently? | Keep kernels constant while changing the transport path |
| When is a specialized stencil worthwhile? | Compare stencil strategies, sizes, and compilation amortization |
| Do platforms need different plans? | Repeat on Orin, Thor, and desktop Blackwell when available |

Publish negative results and platform-specific winners.
Do not generalize from one platform to another without measurements.

## Distribution and adoption

| Channel | Priority and deliverable |
| --- | --- |
| GitHub | Primary home for source, benchmarks, issues, discussions, and adoption |
| PyPI | `pip install cuda-preprocess-graph`, clean-install checks, explicit support matrix |
| ROS ecosystem | Indexed package, Lyrical example, launch files, ordinary tooling instructions |
| Hugging Face | Secondary model demo with pinned model revision and exact preprocessing configuration |
| Kaggle | Secondary educational notebook comparing CPU OpenCV, PyTorch, CV-CUDA, and CPG |
| OpenCV/Kornia/ROS upstream | Optional focused contribution that also serves the product |

The actual repository remains `elcruzo/flex-vision` under the user's repository instruction.
Check package-name availability before distribution.
The README must open with measured end-to-end results once those results exist.
Installation speed is a usability goal, not an unmeasured 30-second guarantee.

The brief identifies these leads for later review:

- [Isaac ROS release information](https://nvidia-isaac-ros.github.io/v/release-5.0/releases/index.html).
- [Isaac ROS image pipeline](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline).
- [OpenCV issue 29318](https://github.com/opencv/opencv/issues/29318).
- [Kornia issue 3872](https://github.com/kornia/kornia/issues/3872).

These are user-provided research leads. Their current status and applicability are not verified here.
Check them before claiming release timing, platform support, or an available contribution opportunity.

## Scope exclusions and future direction

CPG is not an OpenCV replacement, CV-CUDA replacement, image editor, augmentation framework, CNN convolution library, or collection of demos.
Debayering and rectification can appear at pipeline boundaries without becoming custom MVP operators.
Do not add unrelated filters when the success gate fails.

The long-term goal is a small compiler for vision input pipelines.
Its inputs are camera representation, preprocessing graph, and model requirements.
Its output is an optimized GPU execution plan.
