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
CuPy's allocator may reuse freed allocations. CPG does not yet expose a caller-provided output buffer or memory pool.
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
