# Configurable stencil contract

Status: Python/YAML construction and explicit NumPy/PyTorch references are implemented.
CUDA execution rejects this graph until its backend and hardware gate are implemented.
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
