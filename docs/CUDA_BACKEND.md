# Initial CUDA backend

The first backend executes this fixed graph in one CUDA kernel:

```text
RGB8/BGR8 HWC → letterbox → normalize → FP16/FP32 NCHW
```

It does not implement the remaining MVP operators, batching, or autotuning.
A [fixed TensorRT consumer experiment](TENSORRT_EXPERIMENT.md) exists outside the library API.
The complete scope in REQUIREMENTS.md remains required.

## Install and execute

Use the tested CUDA 13 environment from [experiment 004](experiments/004-cuda-host.md).
Install CuPy on an NVIDIA host with `pip install -e '.[cuda]'`.
Do not install the CUDA extra on the Mac for CPU reference work.

```python
import cupy as cp
import cpg

pipeline = cpg.load_pipeline('examples/local-detector.yaml')
frame = cp.zeros((1080, 1920, 3), dtype=cp.uint8)
tensor = pipeline(frame)
assert tensor.shape == (1, 3, 320, 320)
```

The returned value is a CuPy array. PyTorch can consume it with `torch.from_dlpack(tensor)`.
Input can be a CuPy array or a CUDA tensor that implements the object DLPack protocol, including PyTorch.
Both input and output must use the current CUDA device.
CPU arrays are rejected. Upload camera frames explicitly when they start on the CPU.
Objects that implement only CUDA Array Interface are not supported yet.

Each call allocates one output array and no intermediate image arrays.
CuPy's allocator may reuse freed allocations. An experimental caller-provided FP16 output path passed a fixed TensorRT scenario on RTX 4090.
CPG does not yet provide an execution-context memory pool.
A subsequent call does not overwrite a retained output.

## Stream and ownership contract

Calls use CuPy's current stream and synchronize that stream before returning.
This first version blocks the host until preprocessing completes. It is not the final asynchronous runtime.
Input references remain alive through completion, including imported DLPack owners.
No pixel payload moves to the CPU during the backend call.

For CuPy input, order the producer on the current stream or wait for its event before calling CPG.
CPG cannot discover an arbitrary CuPy producer stream from the array alone.
For a PyTorch input, call CPG while the producer's Torch stream is current, so object DLPack can establish the dependency.
Keep stream owners alive until completion.

```python
import torch

producer = torch.cuda.Stream()
with torch.cuda.stream(producer):
    frame = torch.full((1080, 1920, 3), 114, dtype=torch.uint8, device='cuda')
    tensor = pipeline(frame)
```

Asynchronous dispatch, CUDA Graph capture, arbitrary cross-stream usage, multi-GPU execution, and reusable outputs need further design and tests.
Do not assume these are supported because the first kernel runs on CUDA.

## Numerical semantics

The kernel uses the existing graph geometry and independent reference contract:

- Half-up resized dimensions, with each axis at least one pixel.
- Centered padding, with an odd extra pixel on the bottom or right.
- Half-pixel bilinear sampling, edge replication, and no antialiasing.
- FP32 interpolation and normalization, followed by the requested output conversion.
- RGB output, with explicit BGR channel reversal when requested.
- Byte strides for input, including negative strides in CuPy views.

Coordinate calculations use double precision before conversion to FP32 weights, matching the NumPy oracle.
The compiler option `--fmad=false` preserves the oracle's separate multiply/add rounding.
These are initial correctness choices. Measure their costs before changing them.
The first call compiles through CuPy/NVRTC. Cold compilation is not warm execution latency.

## Plan inspection and validation

```bash
cpg inspect examples/local-detector.yaml --height 1080 --width 1920 --backend cuda
python scripts/gpu_backend_check.py --output benchmark-results/backend-check.json
python scripts/gpu_detector_smoke.py --backend cpg --output benchmark-results/cpg-detector
```

Inspection creates no GPU arrays and launches no kernels.
Its launch and temporary-array fields describe the selected implementation, not profiler measurements.
The default inspection backend remains `reference` for compatibility.

The backend check covers numerical variants, strides, retained output, and a Torch DLPack handoff.
The detector runner compares actual model outputs and source-space object boxes against the independent reference.
Validation uploads and downloads are explicit and occur outside the backend's residency claim.
Use an Nsight trace to check actual launches and transfers within the `cpg_preprocess` range.

The [fixed FP16 YOLO comparison](experiments/020-fp16-yolo.md) now records a preprocessing improvement on an RTX 4090 through actual TensorRT inference.
Full G1 acceptance, Jetson support, and ROS GPU transport remain pending.
A bounded cache retains immutable launch metadata by pipeline and input shape. It retains no arrays, strides, or stream owners.

## Experimental caller-owned output

Status: local metadata checks and the fixed FP16 TensorRT reuse scenario passed. See [experiment 022](experiments/022-detector-soak.md).
The existing measured default path still allocates a distinct owned output.

```python
output = cp.empty((1, 3, 320, 320), dtype=cp.float32)
result = pipeline(frame, out=output)
assert result is output
```

Use the shape and dtype from the selected pipeline plan.
`out` must be a CuPy array on the current device with exact output shape and dtype.
It must be C-contiguous and aligned to its dtype.
The runtime rejects overlap with the input's conservative byte extent, including negative strides and gaps.
This conservative check can reject disjoint views whose byte extents overlap.
These checks read metadata, not pixel payloads.
The CuPy layout contract is documented in [CuPy's array implementation](https://github.com/cupy/cupy/blob/main/cupy/_core/core.pyx).

The call overwrites the supplied output and synchronizes its preprocessing stream before returning.
The caller must ensure that earlier consumers of that output finished before reuse.
For another consumer stream, wait for its completion event before submitting the next write.
CPG cannot discover arbitrary external consumers or make a reused output immutable.
Keep the output owner alive until every consumer finishes.
No internal workspace, buffer ring, asynchronous handle, or consumer tracking is implied by this API.

The prepared `scripts/yolo_output_reuse.py` scenario alternates two caller-owned slots through real TensorRT inference.
It checks exact FP16 tensors, dense results, detections, retained alternate slots, default output retention, and Torch inputs.
Run it against newly generated controls from the same committed revision:

```bash
python scripts/yolo_output_reuse.py \
  --engine benchmark-results/iteration-022-yolo/measured/yolo.engine \
  --controls benchmark-results/iteration-022-yolo/measured \
  --output benchmark-results/iteration-022-yolo/output-reuse.json
```

The six-fixture FP16 scenario passed on RTX 4090. FP32 destinations and other stream/device configurations remain pending.
Do not infer allocation savings or latency improvements until measured.
Keep the previously frozen detector soak on its default output path.
Experiment 023 measured caller-owned output through matched inference. It removed pool requests but increased preprocessing host p50 about 5.2%.
Keep it optional. Fresh owned output remains the default. See [the result](experiments/023-output-reuse.md).

## Next stream validation

Read [the asynchronous contract](ASYNC_EXECUTION.md) before changing completion behavior.
`scripts/yolo_stream_ownership.py` prepares 48 inference executions with explicit three-stream handoffs and retained-output checks.
Experiment 024 passed all 48 executions on RTX 4090. Public preprocessing and the fixed TensorRT harness still synchronize before returning.
See [the evidence](experiments/024-detector-streams.md). Asynchronous ownership and overlap remain unvalidated.

## Experimental submission

`Pipeline.submit(frame, stream=...)` returns an explicit preprocessing completion object.
The owned-output path passed 48 fixed TensorRT executions on RTX 4090. Read [experiment 025](experiments/025-async-submission.md) and [the contract](ASYNC_EXECUTION.md).
The synchronous call and out= path retain their existing behavior.
Do not claim async inference, reusable slots, or performance gains from local lifecycle checks.
