# Configurable stencil contract

Status: Python/YAML construction and explicit NumPy/PyTorch references are implemented.
An experimental specialized CUDA backend passed the fixed RTX 4090 v2 gate in experiment 036.
Broader hardware, stream, transport, and performance acceptance remain pending.
This slice advances required custom stencils. It does not complete the operator matrix.

Use one leading conv2d operation followed by letterbox, normalize, and output conversion.
The initial kernel sizes are 3, 5, 7, and 9 on each axis.
The frontend snapshots coefficients as immutable finite FP32 values.
Construction and planning do not execute image operations or allocate GPU workspace.

Convolution flips coefficients on both axes. It applies the same kernel independently to each channel.
For coefficient K[y,x], sample the source at output position plus radius minus (y,x).
Clamp sample coordinates to the original image bounds. Do not introduce zero padding.
The NumPy oracle multiplies and accumulates separately in row-major coefficient order with FP32 arithmetic.
PyTorch uses flipped coefficients with grouped conv2d and replicate padding as an independent reference.
Its accumulation order can differ. Numerical comparisons require a declared tolerance.
No implicit clamp or uint8 rounding occurs after convolution.
Filtering precedes resize and normalization. The later letterbox padding value is not filtered.

```python
pipeline = (
    cpg.Pipeline(input_encoding="bgr8")
    .conv2d([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    .letterbox(640, 640)
    .normalize([0, 0, 0], [1, 1, 1], scale=1/255)
    .to(dtype="float16", layout="NCHW")
)
```

The YAML convolution entry accepts literal rows or a kernel-file path.
Resolve relative paths against the pipeline configuration directory.
Kernel files contain a single kernel mapping with coefficient rows and have a 64 KiB limit.
Use the existing data-only loader and reject duplicate or unknown keys.
Loaded graphs retain coefficient values, not live file references.

The new source is independent implementation code. It does not import cuda-conv source.
The audited flipped-convolution and border behavior inform this contract.
Project licensing remains a separate decision before source import or distribution.

The local inference gate is scripts/local_stencil_reference.py.
It runs all four sizes through independent tensors and the pinned MobileNet classifier.
It also compares literal and relative kernel-file descriptions.
CPU/MPS preprocessing and CPU classifier inference are explicit reference paths.
Neither local reference execution nor metadata tests establish CUDA support or performance.

Primary API reference: [PyTorch functional conv2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.conv2d.html).
CUDA specialization, unrolling, shared-memory/vendor comparisons, stream ownership, and complete-inference transfer capture remain required next work.

Experiment 033 passed all four sizes through CPU and MPS reference preprocessing and real CPU classifier inference.
See experiments/033-stencil-reference.md for errors, source identities, and exact scope. CUDA acceptance remains pending.

## Experimental specialized CUDA path

The planner now exposes two passes and one FP32 HWC temporary for a leading stencil.
The direct stencil source embeds canonical coefficients and emits separate multiply/add statements without runtime coefficient loops.
It preserves coefficient-row accumulation order and disables fused multiply-add.
The second pass fuses letterbox, normalization, output conversion, and NCHW layout.
Original uint8 byte strides feed the stencil. The FP32 temporary uses element strides for its float-pointer consumer.

Both passes use the same preprocessing stream. Submissions retain the original input and temporary until completion.
Advertised CAI producer reuse waits for completion of both passes.
An error after stencil dispatch drains the stream before local temporary owners leave.
A cache retains at most 32 specialized kernel objects. It does not retain frame buffers.
The output remains freshly owned or uses the existing synchronous out= contract.
No reusable workspace, asynchronous out=, shared-memory strategy, automatic tuning, or performance claim is established by this implementation.

The prepared GPU gate is scripts/run_yolo_stencil_experiment.sh.
Its current v2 protocol prepares 288 TensorRT executions: six fixtures, four sizes, positive/negative asymmetric and dense Gaussian kernels, two layouts, and sync/submit.
The original v1 protocol contained 192 executions and remains readable as historical evidence.
It checks exact FP16 tensor bits against NumPy, dense inference, detections, retained outputs, and drainage.
The report verifier rejects incomplete coverage and wrong implementation metadata.
Artificially pending reads, external producers, broader coefficients/dtypes, transfer capture, and credible stencil pipeline timing remain additional gates.
Reconcile the original $15 budget before dispatch. No paid resource is created by preparing this gate.

Primary compilation API checked: [CuPy 14.2 RawKernel](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.RawKernel.html).
Local source-generation and ownership fault tests do not establish CUDA compiler or GPU behavior.

## Prepared matched timing gate

Experiment 034 ran no stencil checks because the GPU container did not expose SSH before the test window became insufficient.
Both pods and the temporary credential are gone. See experiments/034-stencil-startup.md.

After CUDA correctness acceptance, use scripts/yolo_stencil_comparison.py with the same fixed FP16 TensorRT engine.
The required --acceptance report must pass its complete protocol and match the measured source revision and engine hash.
Protocol v1 covers 192 asymmetric cases; v2 adds 96 Gaussian cases. Neither protocol has passed on GPU yet.
Retain that acceptance report with the timing bundle.
The prepared comparison uses both resident 1080p photographs and all four positive asymmetric stencil sizes.
The baseline uses grouped PyTorch FP32 convolution with flipped coefficients and replicated borders.
It then runs bilinear resize, padding, normalization, FP16 conversion, and contiguous NCHW output.
Constants are uploaded before sampling. Both paths return owned outputs and synchronize preprocessing.
No input pixel upload occurs inside the measured interval.

Before sampling, CPG must match exact NumPy FP16 bits.
PyTorch must match the existing 0.0005 tensor absolute tolerance with zero relative tolerance.
Both must pass the existing same-engine dense and detection checks.
Every timed inference repeats those model checks after its measured interval.
That validation can affect the following memory state; report this limit when interpreting results.
The experiment disables TF32 and cuDNN autotuning, warms each path, and alternates path order across an even number of blocks.
It writes per-frame preprocessing, inference, decode, GPU total, and complete host intervals.
The default protocol records 16,000 completed inference checks.

Use scripts/summarize_yolo_stencil.py to reject missing or duplicated coverage and invalid timing values.
The summary calculates p50, p95, and p99 from exported raw samples.
Synthetic verifier fault checks do not establish GPU execution or performance.
No measured stencil comparison exists yet.
This candidate does not establish vendor stencil superiority, reusable workspace, sustained multi-camera throughput, or residency traces.
Compare shared-memory and vendor strategies only after this numerical and complete-pipeline evidence exists.

## Gaussian graph construction

Pipeline.gaussian(size, sigma=...) constructs a leading normalized Gaussian stencil for sizes 3, 5, 7, and 9.
Sigma must be positive finite numeric data. No automatic sigma is selected.
Compute one-dimensional exp(-0.5 * (distance/sigma)^2) weights in Python float arithmetic and normalize their sum.
Form their outer product, then snapshot each coefficient as finite FP32 through the existing convolution contract.
The FP32 sum can differ slightly from one. Preserve FP32 intermediates and replicated borders without implicit clamping.
The configuration accepts a leading gaussian mapping with size and sigma.
Pipeline.custom_conv is an alias of conv2d and uses the same flipped-convolution semantics.

Only one leading stencil is supported. A Gaussian followed by another convolution is not yet a supported graph.
The mathematical Gaussian is separable. The current candidate executes a full specialized two-dimensional stencil.
A separable implementation remains required strategy work and must preserve the declared numerical policy or use measured tolerances.
These frontend additions do not establish Gaussian CUDA acceptance or performance.
Use local_stencil_reference.py --kernel-kind gaussian for the complete local classifier reference check.

Experiment 035 passed all four Gaussian sizes through CPU and MPS reference preprocessing and real CPU classifier inference.
See experiments/035-gaussian-reference.md for identities, measured errors, and the retained sandbox MPS failure.
This accepts the fixed local reference case, not the experimental CUDA Gaussian path or separable strategy.

The prepared v2 verifier requires sigma 1.2 and exact canonical FP32 Gaussian coefficients in each declared plan.
It rejects Gaussian claims with only v1 coverage, wrong coefficients, or a different sigma.
These report checks do not independently recompute model results or establish CUDA execution.
Do not reuse a historical lease reservation for a longer test. Confirm setup time, export allowance, and the remaining original budget before dispatch.

Experiment 036 passed the complete 288-case CUDA v2 gate on RTX 4090, including dense Gaussian coefficients.
It also passed a 6,400-frame matched asymmetric-stencil inference comparison against grouped PyTorch.
See experiments/036-stencil-cuda.md for numerical checks, performance scope, cleanup, and remaining transfer/strategy gates.
Earlier statements of pending hardware acceptance describe preparation before this experiment.
Broad platforms, external-producer delays, Gaussian timing, separable strategies, and stencil transfer capture remain pending.
