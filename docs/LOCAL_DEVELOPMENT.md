# First local development slice

Available: immutable detector graph, plan inspection, NumPy oracle, and explicit CPU/MPS PyTorch reference execution.
No CUDA backend exists yet. Calling `pipeline(frame)` raises an explicit error rather than silently selecting CPU execution.
This slice supports only letterbox → normalize → output conversion.
YAML, other operators, batches, GPU input protocols, and TensorRT remain pending.

## Reproduce the environment

From the repository root, create a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python examples/detector_reference.py
```

The first development environment inherited unrelated system dependency conflicts.
A fresh isolated venv replaced it, and dependency checks plus the complete local inference runs passed.
The pinned file records all installed runtime/test dependencies for this Mac experiment.
No dependency or tool installation modifies the system Python as part of this implementation.

## Numerical contract

Input is a NumPy uint8 HWC image with three RGB or BGR channels.
The reference functions reject other array types rather than coercing device objects into host arrays.
Output is contiguous RGB NCHW with a leading batch dimension of one.

Letterbox preserves aspect ratio, rounds resized dimensions half up, and retains at least one pixel on each axis.
Odd excess padding goes to the bottom or right.
Bilinear sampling uses half-pixel coordinates, edge replication, and no antialiasing.
Interpolation and normalization use FP32 arithmetic before final FP16 or FP32 conversion.
Padding is in the original 0–255 intensity domain and receives the same normalization as image pixels.
Normalization is `(pixel * scale - mean[channel]) / std[channel]` with explicit scale and positive standard deviations.

Geometry records the actual rounded width and height.
Box mapping uses separate actual x/y scales and removes padding before clipping to the source dimensions.
Boxes use continuous xyxy edge coordinates.

## Full local inference check

The initial consumer is TorchVision SSDLite320 MobileNetV3 Large with COCO_V1 weights.
It is a small local harness consumer, not a substitute for the required YOLO/TensorRT benchmark.
The pinned upstream URL and full checksum identify the downloaded weights.

TorchVision code uses BSD-3-Clause.
Its documentation warns that pretrained weights may have additional terms associated with training data.
This experiment uses the weights locally and does not redistribute them or claim unrestricted commercial rights.
Check the model and dataset terms before distribution or product deployment.
Sources: [TorchVision license](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE), [weight documentation](https://docs.pytorch.org/vision/0.23/models.html).

Download the weights into the ignored cache:

```bash
mkdir -p .cache/models
curl -fL --retry 2 https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth -o .cache/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth
.venv/bin/python scripts/local_detector.py --device cpu --output benchmark-results/my-cpu-run
.venv/bin/python scripts/local_detector.py --device mps --output benchmark-results/my-mps-run
```

The runner verifies the complete SHA256 before loading weights with `weights_only=True`.
It refuses to overwrite existing run directories.
MPS is optional and fails explicitly when unavailable.

The runner compares independent NumPy bilinear sampling with PyTorch interpolation on three synthetic image shapes.
Both paths feed the same pretrained CPU detector.
The detector retains its trained internal normalization. External preprocessing applies only the 1/255 scale before the model.
The external 320×320 letterbox is a harness fixture contract, not a claim of canonical SSD accuracy preprocessing.

Dense classification and box-regression outputs must agree even if the synthetic image produces no confident detections.
Postprocessed detections above confidence 0.5 must also agree.
The MPS run intentionally downloads the preprocessing result for comparison and CPU inference.
This path makes no GPU-residency claim.

The runner writes an environment manifest, input hashes, numerical errors, model-output comparisons, and a report.
It records diagnostic elapsed time, not benchmark percentiles or speedup claims.
This synthetic run does not measure detection accuracy, real-camera behavior, or the CUDA success gate.

## Next iteration

Add a licensed representative recorded-image fixture and model-output expectations.
Add Python/YAML equivalence and expand the input contract only as needed for that workload.
Then execute the smallest CUDA/TensorRT path on NVIDIA hardware using the same fixtures and numerical contract.
G0/G1 remain incomplete until the required GPU and inference evidence exists.
