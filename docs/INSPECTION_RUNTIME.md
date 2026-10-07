# Experimental inspection runtime

`cpg.InspectionPipeline` exposes the fixed inspection recipe separately from the detector `cpg.Pipeline` grammar.
Construction and planning do not allocate tensors or execute image operations.
The fixed recipe uses BGR8 input, Gaussian 5×5 with sigma 1.2, sharpen, clamp, crop, and 224×224 bilinear resize.
It produces RGB FP32 NCHW with scale 1/255 and ImageNet mean/std.
See [the numerical reference](INSPECTION_REFERENCE.md) for coefficients, border rules, and model constraints.

```python
import cpg

pipeline = cpg.InspectionPipeline(
    roi=(567, 180, 2706, 1800), strategy="roi"
)
plan = pipeline.plan((2160, 3840, 3))
print(plan.work_box)          # (564, 177, 2712, 1806), x/y/width/height
print(plan.processed_pixels)  # 4897872

# CUDA execution requires a resident PyTorch uint8 HWC BGR tensor.
# Configure cuDNN TF32 off and disable autocast before this fixed FP32 path.
prepared = pipeline.prepare(frame.shape, device=frame.device)
tensor = prepared(frame)
```

The frozen plan records input shape, ROI, strategy, work rectangle, and selection reason.
`strategy="full"` is the default. It processes the full frame before the final crop.
`strategy="roi"` processes the crop and its three-pixel filter halo, clipped to original image bounds.
Both strategies preserve stage order. They do not combine Gaussian and sharpen coefficients.

Selection is explicit. Experiment 014 supports ROI execution for its 4K fixture, but small-image preprocessing was slower.
Those three sizes do not justify a portable automatic threshold.
The API does not claim autotuning or automatically reuse experiment 014 as a performance cache.

## Memory, streams, and precision

Preparation allocates constants once on the requested device.
Execution requires the prepared shape and device and allocates a fresh output for each call.
The implementation does not reuse a caller-owned output or provide a bounded workspace pool.

CUDA operations enqueue on the current PyTorch stream. Execution does not synchronize the host.
Keep the prepared object and input storage alive until queued operations complete.
Order preparation, frame production, execution, and inference on the same stream for the initial supported usage.
Cross-stream use requires caller-managed events, storage lifetimes, and dependencies. This integration has no cross-stream acceptance evidence yet.
Experiment 016 recorded no copies inside the integrated preprocessing-to-classifier ranges on an L4.
That evidence excludes preparation and uploads and does not establish arbitrary cross-stream residency.

Autocast is rejected. CUDA execution also requires cuDNN TF32 to be disabled under the tested PyTorch precision controls.
The runtime does not change process-wide precision settings.
Read [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html) before configuring precision or stream boundaries.
Do not mix old and new TF32 control APIs. Migration of the existing legacy precision policy needs a separate validated change.

CPU and MPS require explicit reference execution:

```python
prepared = pipeline.prepare(frame.shape, device=frame.device, reference=True)
```

This is a local validation facility, not silent GPU fallback.
CuPy, DLPack producers, TensorRT, batched input, YAML inspection graphs, and generic configurable stencils remain unsupported by this fixed API.

## Validation status

The integrated full and ROI paths passed real CPU classifier checks on all three fixtures.
The integrated ROI path also passed MPS preprocessing followed by CPU classifier inference on all three fixtures.
The first MPS attempt exposed a device alias mismatch. Preparation now records the actual allocated device.
The corrected run passed without changing numerical tolerances.
The repository checks passed: 57 tests, including plan validation, edge crops, retained outputs, and autocast rejection.

```bash
.venv/bin/python scripts/local_classifier.py --device cpu --implementation planned-full --output benchmark-results/planned-full
.venv/bin/python scripts/local_classifier.py --device cpu --implementation planned-roi --output benchmark-results/planned-roi
.venv/bin/python scripts/local_classifier.py --device mps --implementation planned-roi --output benchmark-results/planned-mps
```

Each report records the integrated runtime source hash and each fixture's execution plan.
The original NumPy and experimental PyTorch references remain separate from the integrated runtime.
Experiment 014 measured the earlier candidate, not this integration.
The subsequent [integrated CUDA experiment](experiments/016-integrated-inspection.md) passed correctness, matched ROI timing, and transfer traces on an L4.
It measured approximately 42% lower 4K preprocessing latency and 37% lower complete-host latency against the independent baseline.
Small-image preprocessing was slower. Strategy selection remains explicit.
