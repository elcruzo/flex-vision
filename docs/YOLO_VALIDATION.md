# Fixed YOLO validation protocol

## Outcome and boundaries

Validate the required 1080p camera representation → 640×640 FP16 RGB NCHW → YOLO → TensorRT path.
Compare complete inference and preprocessing separately against PyTorch and CV-CUDA on the same GPU.
This protocol does not complete ROS transport, live cameras, Jetson, batching, or detector soak acceptance.

Use YOLOv8n with Ultralytics 8.3.200 and the checksum in `scripts/yolo_common.py`.
Keep weights, ONNX exports, and TensorRT engines outside version control.
Ultralytics supplies this model under its [AGPL/enterprise licensing terms](https://www.ultralytics.com/license).
These experimental dependencies do not establish a license for CPG or authorize redistribution of model artifacts.
No LLM, hosted model API, or training is required.

## Numerical contract

`examples/yolo.yaml` specifies uint8 BGR input, centered letterbox with value 114, RGB conversion, scale 1/255, and final FP16 conversion.
CPG uses half-pixel bilinear interpolation without antialiasing, edge replication, and half-up dimension rounding.
It computes interpolation and normalization in FP32 before the final FP16 conversion.
The independent NumPy oracle defines the expected tensor.

Ultralytics' standard image loader resizes uint8 images through OpenCV.
That intermediate rounding differs from CPG's floating-point interpolation.
Compare this native path separately through real detections. Do not claim pixel equivalence.
Fixed 640×640 output uses `auto=False` in native letterbox. Rectangular dynamic output is outside this experiment.

Freeze these checks before GPU measurement:

- Preprocessing: maximum absolute error 0.0005 against the FP16 NumPy oracle, with zero relative tolerance.
- Dense box output: absolute tolerance 4 pixels plus relative tolerance 0.01.
- Dense class scores: absolute tolerance 0.03 plus relative tolerance 0.03.
- All dense outputs must be finite.
- Detections at confidence 0.5: equal counts, one-to-one class matches, box IoU at least 0.95, score difference at most 0.03.
- Each photo must also detect its annotated object at confidence at least 0.5 and source-coordinate IoU at least 0.5.

The dense limits allow FP16 inference variation. The tighter detection checks constrain meaningful output changes.
Retain failures before changing a limit. Smoke fixtures cannot establish dataset accuracy or task-level recall.
NMS is class-aware, with IoU threshold 0.7 and at most 300 outputs.
TensorRT runs the dense network. TorchVision runs NMS on CUDA for every compared path.

## Feedback loop

1. Run CPU inference on original, odd-padded, and derived 1080p person/cat fixtures.
2. Compare independent NumPy and PyTorch preprocessing through the same trained network.
3. Export a fixed FP16 network and build a strongly typed TensorRT engine on the target GPU.
4. Compare CPG, PyTorch, and CV-CUDA tensors, dense outputs, detections, and source-coordinate boxes.
5. Check CuPy input, PyTorch DLPack input, non-contiguous input, and retained output ownership.
6. Measure two resident 1080p fixtures with the same engine, stream, postprocessing, and synchronization boundary.
7. Warm each path. Rotate candidate order across ten blocks of 1,000 samples per candidate and fixture.
8. Save raw samples, p50/p95/p99, throughput, versions, hashes, engine I/O types, and failures.
9. Capture Nsight transfer evidence separately from latency measurement, with a known transfer control.
10. Export and verify artifacts before independent cloud cleanup. Reconcile actual rental costs.

CPU preprocessing and artifact validation transfers are outside the resident GPU timing range.
Label fresh output allocations and synchronous boundaries explicitly.
Do not infer launch counts, peak memory, power, or bandwidth from operator counts.
Report unavailable metrics as unmeasured.
The 30-minute inspection soak does not substitute for a detector soak.

## Baseline choices and primary sources

The PyTorch baseline uses floating-point resize, padding, scaling, and layout conversion.
The vendor candidate uses CV-CUDA resize/crop/convert/reformat with `srcCast=False`, channel reversal, scale, and FP16 output.
It then pads and reformats to NCHW. Verify this candidate against the same numerical oracle before timing it.
An unsupported or incorrect vendor candidate must remain a reported limitation, not a silently removed comparison.

- [Pinned Ultralytics predictor](https://github.com/ultralytics/ultralytics/blob/v8.3.200/ultralytics/engine/predictor.py)
- [Pinned letterbox implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.200/ultralytics/data/augment.py)
- [CV-CUDA operators](https://cvcuda.github.io/CV-CUDA/modules/python/operators.html)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html)

The local reference checks passed all six photo cases at source revision `d09b809` with the new experiment files present.
The maximum NumPy/PyTorch tensor difference was 0.00048828125. Both 1080p cases had identical tensors.
The native Ultralytics path also detected each expected object. Its tensor differences remain diagnostic.
See the retained [local results](experiments/data/020/local/results.json) and [environment](experiments/data/020/local/environment.txt).
The first local invocation completed inference but failed during environment export because `python` was absent from PATH.
The corrected runner uses its own interpreter. The second invocation completed and saved its environment.

At committed revision `ad1fef0`, CPU FP16 model inference also passed all six cases against the FP32 model.
The minimum matched detection IoU was 0.99857. Dense differences remained within the declared limits.
The CPU ONNX export passed graph validation with FP16 input and output, shapes `[1,3,640,640]` and `[1,84,8400]`.
These checks do not replace TensorRT execution on NVIDIA hardware.
See the [FP16 results](experiments/data/020/local/fp16-results.json) and [export record](experiments/data/020/local/onnx-results.json).

## GPU failure and model preparation correction

The first GPU execution built the FP16 TensorRT engine, then failed its same-input dense comparison.
One of 33,600 box coordinates exceeded the original limits. Its absolute difference was 9.5 pixels.
No performance sample was collected. The failure and export checks remain in [the initial GPU record](experiments/data/020/initial-gpu/results.json).
All ten artifacts were verified before the GPU, controller, and temporary credential were removed.

The original PyTorch reference retained batch normalization, while ONNX export folded it into FP16 convolution weights.
The pinned upstream exporter first fuses convolution and batch normalization in FP32, then converts the model to FP16.
Use that same preparation for the reference and export. The [upstream source](https://github.com/ultralytics/ultralytics/blob/v8.3.200/ultralytics/engine/exporter.py) specifies this order.
This is a correction to model preparation, not a numerical tolerance change.
Compare fused and unfused FP32 outputs locally, then rerun all GPU checks with the original limits.
Save control tensors before assertions so subsequent failures retain the actual dense values.

## Acceptance revision 2: separate preprocessing from engine variation

The fused-model GPU retry passed both person cases, including the vendor baseline and GPU framework imports.
It failed the cat case on 12 background box coordinates in the PyTorch/TensorRT comparison.
Those proposals scored below 0.000001 in both engines. The actual cat detections had identical confidence and box IoU 0.99801.
At score 0.01 or above, 24 proposals remained and their maximum box difference was one pixel.
The original global box-coordinate gate therefore remains failed. Keep [that failure](experiments/data/020/fused-gpu-failure/results.json).

Use these explicitly revised acceptance rules for the next run:

- Compare CPG and both preprocessing baselines against the NumPy tensor passed through the **same TensorRT engine**.
- Keep the original, unmasked dense box and score limits for those same-engine comparisons.
- For PyTorch/TensorRT cross-engine comparison, check every dense score, every output's finiteness, and valid probabilities and box dimensions.
- Apply the original box tolerances to proposals with score at least 0.01 in either engine.
- Report original-limit violations on lower-confidence background boxes. Do not conceal or call those coordinates equivalent.
- Preserve the unchanged one-to-one detection checks at confidence 0.5, source-box checks, and tensor tolerance.

The 0.01 floor is 50 times below the final detection threshold. It protects near-threshold proposals while separating irrelevant background coordinates.
This is an evidence-driven acceptance change, not a claim that revision 1 passed or that all FP16 engine outputs are equivalent.
The revised policy still needs GPU execution. Smoke fixtures do not establish dataset accuracy.

## Acceptance revision 3: preserve background drift for every path

Revision 2 passed the original person fixture through all paths. On the padded person fixture, CPG passed but the PyTorch baseline failed.
Its tensor remained within the declared FP16 tolerance. One background width exceeded the global dense-box limit by changing seven pixels.
The independent reference score at that anchor was about 0.00014. The [baseline failure](experiments/data/020/baseline-failure/results.json) remains failed under revision 2.

Use the same explicit 0.01 proposal-score floor for box checks in preprocessing comparisons.
Keep every background difference and original-limit violation in the report.
Tighten every same-engine dense score comparison to absolute 0.003 plus relative 0.003, from the earlier 0.03 plus 0.03.
Keep the original tensor tolerance and final detection count, class, score, and coordinate checks.
Require every output to remain finite, every box dimension nonnegative, and every probability in [0,1].
This protocol validates inference behavior on smoke fixtures. It does not establish equality of low-confidence background coordinates.

The next vendor baseline uses two passes: resize/channel conversion/scale/FP16/NCHW, then padding directly into an owned NCHW output.
This removes the earlier separate layout pass. Verify this plan before measuring it.
The earlier three-pass vendor results remain in the failed-run records, not as a claim about the fastest vendor plan.
