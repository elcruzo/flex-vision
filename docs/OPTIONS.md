# Design possibilities and scope preservation

This register preserves ideas from the CPG brief and its earlier FluxVision direction.
It does not turn every possibility into a first-release requirement.
It also does not discard a possibility merely because its implementation needs evidence.
[REQUIREMENTS.md](REQUIREMENTS.md) defines required CPG outcomes.

## How to treat an idea

Use four distinct statuses:

- Required: deliver the stated capability and attach acceptance evidence.
- Candidate: compare this implementation with alternatives for a required outcome.
- Later option: preserve the idea and its trigger for investigation.
- Unverified premise: check the external claim before using it to make a decision.

“Pending hardware” describes missing evidence. It does not change required scope.
“Candidate rejected” describes one implementation. It does not cancel its required outcome.
Do not replace a required feature with a placeholder and call its milestone complete.

## Complete camera boundary

Retain the complete motivating path when examining bottlenecks:

```text
camera → debayer/color conversion → rectify → denoise → sharpen
       → resize → letterbox → uint8-to-float → normalize
       → HWC-to-CHW → TensorRT
```

A particular model does not necessarily need every stage.
Measure the stages it actually uses and disclose external stages.
Debayering and rectification remain integration possibilities even though custom implementations are outside the initial operator list.

The intended progression remains:

```text
cuda-conv → generic custom stencil → multiple operations → pipeline graph
          → fusion and autotuning → ROS CUDA buffers
          → camera-to-TensorRT evidence → robotics and industrial use
```

The graph can exist early without postponing stencil capability to an unrelated future project.
Use a minimal detector path to establish the harness, then test the custom-stencil inspection path.

## Optimization candidates

| Candidate | Why preserve it | Experiment or decision trigger |
| --- | --- | --- |
| Naive stencil | Simple reference GPU implementation from cuda-conv | Correctness and whole-pipeline baseline |
| Shared-memory tiles | Reuse neighborhoods within a block | Compare traffic, occupancy, borders, and total latency |
| Specialized 3×3/5×5/7×7/9×9 | Exploit known stencil sizes and coefficients | Generated code inspection and end-to-end timings |
| Constant coefficients and loop unrolling | Avoid generic coefficient/loop overhead | Compare compilation cost, code size, and warm execution |
| Separable Gaussian | Reduce work for separable kernels | Compare two passes with direct or fused alternatives |
| Vectorized loads and channel packing | Improve access patterns | Alignment, stride, channel-count, and throughput checks |
| Gaussian plus sharpen | Reduce intermediate image traffic | Border and rounding equivalence before performance measurement |
| Stencil plus scale/clamp/normalize | Avoid full-image intermediate passes | Verify operation order and numeric range |
| Resize/letterbox plus normalize/type/layout | Optimize the detector boundary | E01 comparison with real inference |
| Memory pool and ping-pong workspace | Reuse storage between stages and frames | Steady-state allocation and lifetime checks |
| Batched kernels | Avoid Python loops per image/channel | Batch latency, throughput, and inference compatibility |
| Concurrent CUDA streams | Overlap streams, transfers, and inference | Fairness, dependencies, contention, and p99 |
| Generated CUDA versus vendor operations | Select the fastest valid complete plan | CV-CUDA/VPI/internal/CuPy candidate comparisons |
| Autotuning and persistent plans | Adapt strategy to the current GPU and workload | Tune/restart/cache-hit end-to-end test |
| CUDA Graph capture | Reduce repeat dispatch overhead where compatible | Additional candidate beyond the original PRD, measured separately |

No backend must win every operator.
Include intermediate conversions, allocations, and synchronization when evaluating a vendor operation inside a complete plan.
Do not choose an individually faster primitive that slows the full workload.

## Preserved FluxVision possibilities

The newer CPG brief controls the first release where the briefs differ.
These earlier possibilities remain visible rather than disappearing from the roadmap.

| Earlier possibility | Current treatment | Trigger for investigation |
| --- | --- | --- |
| Grayscale and RGBA | Later input coverage beyond the initial RGB/BGR focus | Real customer/model requirement and numerical tests |
| H×W, H×W×C, N×H×W, N×C×H×W | Explicit layout extensions, never shape guessing | Versioned input contract and batch tests |
| uint8, FP16, FP32 input | Preserve dtype coverage planning per operator | Reference semantics and backend support matrix |
| Laplacian and unsharp mask | Later operators, not silently substituted for CPG requirements | Workload benefit after the two-pipeline gate |
| Morphology, thresholding, color transforms | Later scope | A concrete pipeline bottleneck or integration need |
| Gaussian separability | Active optimization candidate | Whole-graph comparison, not operation-count reasoning alone |
| ONNX Runtime and DeepStream | Later inference integrations | User demand and zero-copy consumer proof |
| JAX, TensorFlow, RAPIDS | Possible DLPack consumers | Verified ownership/stream contracts and actual workloads |
| C++ API and stable ABI | Later deployment option | Consumers that cannot use the Python runtime |
| C++ planner/pool/dispatcher | Possible runtime architecture | Measured Python overhead or host integration requirements |
| Architecture-specific binary wheels | Distribution option alongside runtime specialization | Build capacity, target tests, startup and installation requirements |
| Webcam/video inference demo | Useful input adapter and demo extension | E01 harness and available capture hardware |
| Industrial defect demo | Required inspection workload, model-specific demo later | Representative dataset and classifier |
| OCR preprocessing demo | Later showcase, including document-page throughput | Workload evidence after core gates |
| GPU cost per processed frame | Later product metric | Sustained runs with explicit compute and operating cost assumptions |
| Stars, downloads, integrations, repeat users | Adoption observations | Public releases and voluntary user feedback |

Do not claim a reduction in required GPUs from a single isolated-kernel measurement.
Measure sustained completed work, model quality, load, and hardware cost before making that claim.

## Research and ecosystem possibilities

Preserve the six-camera humanoid and six-4K-camera factory scenarios as workload motivations.
The illustrative missing 9 ms is a customer example, not a measured CPG result.
The “weird 20%” describes custom preprocessing needs, not a measured market share.

Hugging Face demo candidates include a small YOLO, RT-DETR, or segmentation model.
Pin the model revision, input requirements, preprocessing configuration, and redistribution permissions.
A Kaggle notebook can compare CPU OpenCV, PyTorch, CV-CUDA, and CPG for education.
Keep CPU comparisons separate from the GPU success gate.

Preserve OpenCV runtime compilation/fusion and Kornia GPU-batched letterbox as upstream investigation leads.
Check the linked issues before deciding whether they remain open or suitable.
A focused contribution can support adoption, but it must not block the core runtime.
The older five-star count and stated Isaac ROS release date are historical brief claims, not maintained project facts.

DGX Spark, desktop RTX, and Blackwell comparisons can help explain platform-specific winners.
Retain Orin, Thor, and desktop Blackwell as the named cross-platform research question.
Use vendor transport and rectification as serious alternatives in those experiments.

## Reports and demo presentation

Preserve the information hierarchy of the PRD's sample profiler:

```text
Device and input representation
Original graph: per-stage measurements
Original allocations, launches, host copies
Optimized graph: per-group measurements
Optimized allocations, launches, host copies
Measured total
```

Candidate groups include Gaussian plus sharpen and letterbox plus normalize/type/layout. Their fusion still requires validation.
Render unsupported fusion as separate groups with a reason.
Keep the PRD's timing numbers out of real reports because they are illustrative.

The README comparison should show baseline versus CPG p50/p99, copies, and launches for the camera-to-model boundary.
The Jetson demo should show camera resolution/FPS, preprocessing, inference, total, CPU copies, GPU memory, and dropped frames.
A target such as 1920×1080 at 60 FPS needs an actual run before becoming a supported performance claim.
Include an inspectable Nsight trace with the launch demo, not only a dashboard screenshot.
