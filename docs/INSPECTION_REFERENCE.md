# Inspection workload reference

This fixed workload prepares the second GPU experiment through real classifier inference.
It executes on the Mac today. It does not add production Gaussian, sharpen, crop, or resize operators to `cpg.Pipeline`.

```text
BGR8 frame
  → RGB FP32
  → Gaussian 5×5, sigma 1.2
  → sharpen 3×3
  → clamp to [0, 255]
  → crop ROI
  → resize 224×224
  → scale and ImageNet normalization
  → FP32 NCHW
  → MobileNetV3 Small classifier
```

## Numerical contract

The Gaussian uses normalized separable coefficients, computed in FP64 and stored as FP32.
Horizontal filtering precedes vertical filtering. Both stages and sharpen use replicated image borders.
Sharpen coefficients are `[[0,-1,0],[-1,5,-1],[0,-1,0]]`.
These fixed symmetric filters do not establish arbitrary custom-convolution semantics.

All intermediate image values remain FP32. Clamp occurs after sharpen. There is no intermediate uint8 quantization.
ROI coordinates are integer x, y, width, and height, with an exclusive right/bottom edge.
Filtering occurs before cropping. A crop boundary therefore receives pixels from outside the ROI where the original image permits it.
Clamping or padding an isolated ROI before filtering would change the contract.

Resize stretches the ROI to 224×224 using bilinear half-pixel sampling, edge replication, and no antialiasing.
Normalization is `(pixel / 255 - mean) / std`, with ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
Output is contiguous FP32 RGB NCHW with batch size one.

This is a defined experimental input contract, not the checkpoint's canonical accuracy-evaluation transform.
TorchVision's documented transform resizes to a 256-pixel short edge and then center-crops to 224×224.
Our ROI pipeline tests preprocessing equivalence and semantic smoke behavior. It does not establish ImageNet or defect-classification accuracy.
[Model documentation for the tested TorchVision release](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.mobilenet_v3_small.html)

## Model and fixtures

The consumer is MobileNetV3 Small with `IMAGENET1K_V1` weights.
Pinned URL: `https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth`.
Full SHA256: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.
The runner verifies the full hash before loading with `weights_only=True`.
Weights remain in the ignored cache. No model training or hosted inference service is required.
The existing TorchVision/model-data licensing qualifications in [local development](LOCAL_DEVELOPMENT.md) also apply here.

The pinned Chelsea photograph supplies three cases:

- Original 300×451 image, with the complete image as the ROI.
- Odd-sized 333×490 canvas, with the original image at x=13, y=11 and an exact ROI.
- 2160×3840 canvas, with sixfold pixel repetition of the photograph and an exact centered ROI.

The last case has ROI x=567, y=180, width=2706, height=1800.
The canvas uses value 114. Both padded cases exercise filtering across the ROI boundary.
These photographs are smoke fixtures, not representative industrial defects or production camera captures.
Read [fixture provenance](../tests/fixtures/images/README.md) for image hashes and rights.

## Run the feedback loop

Use the existing isolated Mac environment from [local development](LOCAL_DEVELOPMENT.md).
Download the model once:

```bash
curl -fL --retry 2 https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth -o .cache/models/mobilenet_v3_small-047dcff4.pth
.venv/bin/python scripts/local_classifier.py --device cpu --output benchmark-results/inspection-cpu
.venv/bin/python scripts/local_classifier.py --device mps --output benchmark-results/inspection-mps
```

Each output directory must be new. MPS must be available when requested.
The runner rejects an explicit MPS CPU-fallback setting.
MPS runs preprocessing only. The harness downloads its tensor for independent comparison and CPU model inference.
Those deliberate transfers do not establish GPU residency.

NumPy filtering uses padded slices and explicit weighted accumulation.
The PyTorch reference uses grouped convolutions and interpolation. Both feed the same real pretrained classifier.
Required checks are:

1. Every normalized tensor element agrees within absolute error 0.0002.
2. All 1,000 logits agree within absolute error 0.001 and relative error 0.0001.
3. The ordered top-five labels match.
4. A cat class appears in the top five, and the five cat classes have combined probability at least 0.2.

The known cat indices are 281 through 285 in the pinned ImageNet category order.
The semantic expectation is independent of the observed top-one prediction.
The runner saves tensor and logit arrays, class names, probabilities, source hashes, recipe coefficients, model identity, and errors.
Elapsed seconds are diagnostic values, not comparative performance samples.

Check that a false semantic expectation fails the complete runner:

```bash
.venv/bin/python scripts/local_classifier.py --negative-control --output benchmark-results/inspection-negative
```

This command must exit nonzero and save a failed report that identifies the absent dog-class expectation.
Do not count that deliberate failure as an ordinary passing inference result.

## Path to the GPU experiment

Keep this reference and its numerical tolerances independent of the eventual backend implementation.
First establish equivalent PyTorch/CuPy CUDA filtering, crop, and classifier inference.
Then measure candidate implementations against that complete baseline.

Candidate optimizations include separable filtering, reusable workspace, stencil specialization, and fused crop/resize/normalization.
Computing only the ROI plus the required filter halo is another candidate, subject to global border and rounding equivalence.
Do not assume that combining Gaussian and sharpen preserves finite-precision results or image-border behavior.

Required later evidence includes actual GPU classifier output, retained buffers, streams, transfer traces, and raw repeated latency samples.
The planned industrial workload still needs representative inspection images and an appropriate classifier or dataset evaluation.
The two-workload performance gate remains open until this second workload has credible GPU results.

## Prepared CUDA baseline runner (hardware validation pending)

`TorchInspectionBaseline` stores filter and normalization constants on the selected device once.
Its `from_bgr8` method accepts a resident PyTorch uint8 HWC BGR frame.
It runs conversion, filtering, crop, resize, and normalization without an explicit pixel download.
The caller must preserve input ownership and current-stream dependencies until execution completes.
This experimental adapter does not extend the public CPG operator catalog.

Run the same classifier checks on NVIDIA hardware:

```bash
python scripts/local_classifier.py --device cuda --output benchmark-results/inspection-cuda
```

The runner uploads the fixture before preprocessing, then feeds the resulting tensor directly to MobileNetV3 Small on CUDA.
It downloads tensors and logits only after inference, for comparison against independent NumPy preprocessing and CPU inference.
The `inspection_to_classifier` NVTX range marks that device boundary, including completion synchronization.
A trace must still verify transfers. The presence of an NVTX range does not prove residency.
The runner disables TF32 for convolution and matrix multiplication and records the GPU and CUDA version.
The precision and stream policy follows [PyTorch 2.9 CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html).

CPU and MPS checks validate shared code locally. Experiment 012 records the corresponding CUDA classifier run, transfer trace, and warmed repeated baseline on an L4.
Do not treat that reference baseline as production CUDA operator support or as evidence that CPG improves the workload.

The CUDA smoke runner performs three warmup passes per fixture before its NVTX range.
Use `--warmup 0` only for an explicitly cold diagnostic. No warmup changes the required numerical tolerances.

After a passing CUDA smoke report at the same clean revision, collect the baseline:

```bash
python scripts/inspection_benchmark.py \
  --validation benchmark-results/inspection-cuda/report.json \
  --output benchmark-results/inspection-latency
```

The benchmark rejects failed, negative-control, stale-source, or mismatched-device validation reports.
It checks model and fixture hashes before measurement.
Each fixture runs four repeats of 300 serial frames, with 30 warmup frames before each repeat.
Raw CSV rows contain preprocessing, inference, complete CUDA-stream intervals, and completed host-call latency.
Percentiles use NumPy's linear method without outlier removal.
Fixture uploads, reference comparisons, and result downloads are outside these intervals.
These measurements do not include camera acquisition, upload latency, or an industrial deployment.

The report includes PyTorch peak allocated bytes and reserved bytes.
Those allocator observations do not measure total GPU memory, bandwidth, power, or kernel launch counts.
Use a separate Nsight run for launch and transfer evidence. Do not mix profiled timing with ordinary benchmark timing.
A passing baseline is not evidence that CPG improves the second workload.
