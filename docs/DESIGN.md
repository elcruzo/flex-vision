# Design notes

Status: proposed contracts. No runtime exists yet.

## Pipeline model

```text
image contract + operation graph + model input contract
                         ↓
                validation and planning
                         ↓
       implementations + buffer lifetimes + stream dependencies
                         ↓
                  GPU execution plan
```

The frontend records operations without executing them.
Validation rejects unsupported types, dimensions, parameter values, and operation order before kernel launch.
The planner must preserve the semantics of the original graph.

A linear graph is sufficient for the first detector workload.
Branching and general scheduling require demonstrated use cases.
Python can own the initial planner. A C++ runtime is a later decision, not an initial requirement.

## Image and tensor contracts

Every input needs shape, dtype, layout, channel encoding, strides, device, ownership, and stream information.
Do not infer a layout from ambiguous dimensions.
Start with HWC input. Add NHWC and NCHW batches through explicit contracts.
Reject unsupported views clearly rather than copying them silently.

Define these rules before each operator enters the runtime:

| Area | Required decision |
| --- | --- |
| Resize | Interpolation, pixel-center coordinates, antialias behavior |
| Letterbox | Aspect-ratio rule, integer rounding, pad split, pad value |
| Crop | Bounds policy, coordinate convention, view or output buffer |
| Normalize | Scale, channel order, mean/std broadcast, zero-std rejection |
| Stencil | Correlation or flipped convolution, border rule, accumulator dtype |
| Conversion | Rounding, saturation, overflow, FP16 error tolerance |
| Sobel | Separate gradients or magnitude, output range and dtype |
| Arithmetic | Broadcasting, operation order, clamp placement |

Proposed normalization is `y[c] = (x[c] * scale - mean[c]) / std[c]`.
Specify scale explicitly in saved configurations.
A proposed single-image NCHW output has shape `[1, C, H, W]`.
Preserve resize and padding metadata so detections can map back to camera coordinates.

## Memory and streams

“No host copy” concerns the pixel payload. Small graph and tensor metadata can reside on the CPU.
“No host copy” does not mean no GPU reads, writes, or allocations.

Retain input owners until asynchronous reads complete.
Retain output and workspace owners until their consumers complete.
A device pointer alone is insufficient to prove safety.

Use DLPack or the CUDA Array Interface only after checking their current stream and lifetime requirements.
Check producer readiness before consuming an input.
Expose consumer readiness without a device-wide synchronization on each frame.
Test non-default streams and cross-framework handoffs explicitly.

Reuse workspace only when its previous work completes.
Use separate execution contexts or event-governed reuse for concurrent streams.
Bound caches and memory pools for changing input shapes.
Provide a caller-owned output path if measurements justify it.

## Fusion and implementations

Start with the resize/letterbox, normalize, conversion, and layout group.
Keep a correct unfused path as a comparison and diagnostic tool.
Do not equate fewer launches with lower end-to-end latency.

Stencil fusion needs special care around borders and intermediate rounding.
Combining Gaussian and sharpen coefficients can change boundary behavior.
Reordering nonlinear operations such as clamp can change output values.
A fusion rule requires reference tests and an explicit numerical policy.

Possible implementations include generated CUDA, internal kernels, CuPy, CV-CUDA, and VPI.
These are candidates, not current dependencies or promised support.
Select them using complete-plan cost and compatibility evidence.

## Configuration and errors

A future YAML schema describes input, ordered operations, and output requirements.
Validate the schema before opening a camera or allocating GPU workspace.
Reject unknown fields and invalid kernels with their configuration location.
Resolve kernel file paths relative to the configuration file.
Treat configuration as data. Do not execute arbitrary Python from YAML.

Cache compiled kernels and tuned plans with versioned keys.
A stale cache must not silently select incompatible code.
Provide a useful error when the requested GPU path is unavailable.
A CPU fallback must be explicit because it changes latency and transfer behavior.

## Interfaces to validate

| Interface | Proof required before claiming support |
| --- | --- |
| CuPy | Native arrays, strides, memory owner, selected stream |
| PyTorch CUDA | DLPack handoff, device agreement, lifetime, consumer stream |
| CUDA Array Interface | Current protocol, stream semantics, owner retention |
| TensorRT | Binding contract, dtype/layout, stream ordering, output lifetime |
| ROS CUDA buffer | Current API, allocator, transport, publisher/subscriber ownership |
| CPU ROS Image | Explicit upload, pinned-memory policy, measured transfer bytes |

## Open decisions

- Which project license permits the intended source reuse and distribution?
- Which GPU is available for repeatable development measurements?
- Which package and runtime versions work together on each requested Jetson target?
- Does direct CUDA generation outperform vendor composition on the selected workloads?
- Does the first release need a compiled host runtime?
- Which tensor message fits the chosen ROS inference consumer?
- Which binary artifacts can CI build and validate on actual hardware?

Record evidence and decisions in [DECISIONS.md](DECISIONS.md).
