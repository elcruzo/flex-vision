# Configurable stencil contract

Status: Python/YAML construction and explicit NumPy/PyTorch references are implemented.
An experimental specialized CUDA backend is implemented. CUDA compilation and hardware acceptance remain pending.
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
It covers 192 actual TensorRT executions: six fixtures, four sizes, positive/negative asymmetric kernels, two layouts, and sync/submit.
It checks exact FP16 tensor bits against NumPy, dense inference, detections, retained outputs, and drainage.
The report verifier rejects incomplete coverage and wrong implementation metadata.
Artificially pending reads, external producers, broader coefficients/dtypes, transfer capture, and credible stencil pipeline timing remain additional gates.
Reconcile the original $15 budget before dispatch. No paid resource is created by preparing this gate.

Primary compilation API checked: [CuPy 14.2 RawKernel](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.RawKernel.html).
Local source-generation and ownership fault tests do not establish CUDA compiler or GPU behavior.
