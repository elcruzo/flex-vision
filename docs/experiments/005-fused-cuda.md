# First fused CPG CUDA execution

Result: the initial detector graph passes numerical, real-inference, and scoped trace checks on an RTX 4090.
This is a correctness milestone, not a performance or full G1 acceptance result.

Backend commit: `ef44f60`. Detector and traced run commit: `c7f1544e4bc8a17ef7e865de4af88f8b081e8ceb`.
The detector manifest records a clean working tree and source hashes.

## What changed

`Pipeline.__call__` now executes a fused CUDA kernel for the existing detector graph.
It combines RGB/BGR conversion, letterbox, normalization, FP16/FP32 conversion, and NCHW output.
Input stays on the GPU. Each call produces a new CuPy output without temporary image arrays.

This version synchronizes its stream before returning.
That makes input lifetime and output readiness explicit, but prevents asynchronous host dispatch.
Read [the backend contract](../CUDA_BACKEND.md) before using it.

## Evidence

- All 43 local checks passed after the API and inspector changes.
- All 48 GPU combinations passed: six shapes, two dtypes, two encodings, and contiguous or negative-stride input.
- Shapes include 1×1, single-row, single-column, odd dimensions, and 1080p input.
- Every combination retained its first output while executing another call. Outputs had distinct live allocations and unchanged retained values.
- The PyTorch CUDA input check passed through object DLPack with a separate producer stream.
- Invalid floating-point input was rejected.
- All four photograph cases passed real SSDLite inference twice, including a traced run.
- Photo input tensors matched the independent NumPy reference exactly in this run.
- Dense model outputs and high-confidence detections passed the existing fixed tolerances.
- `pip check` reported no broken requirements.

| Photograph case | Source-space expected-object IoU |
| --- | --- |
| astronaut | 0.930435 |
| astronaut-padded | 0.863174 |
| chelsea | 0.944010 |
| chelsea-padded | 0.916464 |

These broad smoke annotations do not establish dataset accuracy.
The model and image hashes are unchanged from the earlier experiments.

## Trace acceptance

The four synchronous `cpg_preprocess` NVTX ranges each contain:

- Exactly one CUDA kernel event.
- Zero overlapping CUDA memory-copy events and zero copied bytes.

The independent validation harness performs uploads, baseline inference, and result downloads outside these ranges.
The zero-copy observation applies to this backend scope, not the entire harness or future ROS camera transport.
The trace does not prove that no allocation occurs. Source inspection identifies one output and no intermediate image arrays.

Reproduce the trace check after exporting Nsight's SQLite file:

```bash
python scripts/check_cuda_trace.py benchmark-results/fused-trace.sqlite --expected-ranges 4 --output benchmark-results/trace-validation.json
```

The checker requires synchronous completion within each range.
It must change if the backend becomes asynchronous.
Negative controls rejected an arithmetic-only trace without NVTX events and the earlier unfused trace without CPG ranges.

All 16 remote artifacts matched their SHA256 values after transfer to the Mac.
Local directory: `benchmark-results/iteration-005-fused/`.

| Artifact | SHA256 |
| --- | --- |
| fused-trace.nsys-rep | f37a4aba640a7c754c0b3073fbd88319ed56a182cea9fb6f4ff794cc15221302 |
| fused-trace.sqlite | 96d04820c01a5caa08361790442609ec147f8bb9d7278159cc462c856bfc35e9 |
| fused-traced/report.json | b11f30ed182c87c7d433bfffee4cb70b5133d196070393eddc1bfefc532d3646 |
| backend-check.json | 2d5af6ce4c5888b44cb0c5be8289ad692df70d7e8913e992180fbc1a86d5f5e2 |

Large traces and tensor outputs remain outside Git.

## Environment and cost

The session reused experiment 004's CUDA 13/PyTorch 2.9.1/TorchVision 0.24.1/CuPy 14.2.0 package choices.
The official image was `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`.
Runpod's startup log reported digest `sha256:ab2addc2916ffc72989288bd5048933c69ba6531f1d679c25afbd9eadc5a5fd5`.
The package inventory and hardware preflight are retained with the artifacts.

Pod `hkcvv93vzorup5` ran in EU-RO-1 at the reported $0.74/hour rate, with 50 GB temporary disk.
It started at 08:16:51 UTC and was terminated at approximately 08:21:12 UTC on 2026-10-04.
Estimated session cost is $0.06 including disk. Estimated cumulative spend is approximately $0.16 of the authorized $15.
The provider billing ledger had no posted records at cleanup. Final charges remain pending.

Artifacts were verified before termination. The subsequent account read returned no Pods.
No persistent volume was created. Local and Pod-side timers served as fallback termination controls, not guaranteed provider spending caps.

## Next milestone

Connect the produced device tensor to an actual TensorRT model, then compare complete pipeline latency against an equivalent GPU baseline.
Separate cold compilation, kernel timing, synchronous host cost, and end-to-end model timing.
Do not infer a speedup from one kernel or from the four correctness fixtures.
Asynchronous ownership, reusable output buffers, CUDA Array Interface, batching, Jetson, and ROS acceptance remain pending.
