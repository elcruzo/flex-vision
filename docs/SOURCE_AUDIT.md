# cuda-conv source audit

Reviewed revision: `c93a5b28db3dc6377ba5c13bbb3787c252bf7347`.
Source: [cuda-conv](https://github.com/elcruzo/cuda-conv/tree/c93a5b28db3dc6377ba5c13bbb3787c252bf7347).
This is a source review on the Mac. No upstream CUDA tests ran here.

| Source | Finding | Migration decision |
| --- | --- | --- |
| LICENSE | MIT, with named upstream copyright | Retain notice with any future imported source |
| src/kernels/conv2d.py | FP32 naive and 16×16 shared-memory kernels, maximum stencil size 9 | Retain as candidates for a later GPU experiment |
| src/kernels/conv2d.py | Flipped coefficients and clamp-to-edge borders | Preserve this behavior explicitly when importing |
| src/api.py | Default return_numpy=True copies payload to CPU | Replace the wrapper for the CPG residency contract |
| src/api.py | RGB uses per-channel contiguous copies and a final stack | Compare direct batch/channel dispatch instead |
| src/kernels/conv2d.py | Allocates output for each invocation | Add explicit output ownership and reusable workspace |
| src/timing.py | Kernel-only RGB takes a strided channel view | Repair before reuse: raw kernel assumes contiguous indexing |
| src/timing.py | Warmup and timings use the null stream | Replace with explicit measured-stream semantics |
| src/timing.py | Reports means and CPU-relative speedup | Preserve event-timing technique, replace report with full pipeline evidence |
| tests | Border, asymmetric kernel, dtype, and preset coverage | Retain useful cases, add ownership and full inference scenarios |

No upstream implementation code is imported in this first slice.
The first frontend/reference implementation is new code.
Project licensing remains pending until the maintainer chooses it, before source import or distribution.
Later NVIDIA experiments provide execution evidence. G0 remains incomplete because licensing and broader target validation remain pending.
