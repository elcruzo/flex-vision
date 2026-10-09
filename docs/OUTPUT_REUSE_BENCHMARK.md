# Matched caller-owned output experiment

Hypothesis: avoiding the output allocator request can reduce preprocessing overhead without harming complete inference tails.
The optional output path remains synchronous. This experiment changes output allocation, not stream synchronization or numerical semantics.
No minimum speedup is assumed. Retain a neutral or slower result.

## Fixed comparison

Use the pinned FP16 YOLO/TensorRT engine, two resident 1080p fixtures, and shared CUDA NMS from experiment 022.
Compare the default fresh owned output with one preallocated caller-owned CuPy FP16 output.
Keep that slot allocated throughout both candidates, and retain common pools without cache clearing.
Release per-frame outputs only after inference and validation finish.
The same serial consumer and stream execute both candidates.

Run 100 checked warmups per candidate, then ten blocks of 1,000 samples per candidate and fixture.
Alternate candidate order by block. Retain all 40,000 sampled frames.
Check every sampled tensor, dense output, validity, and final detection on the GPU with the existing CPG policy.
Download the scalar result after stopping the timing interval.
Validation can condition later samples and remains a qualification of this comparison.
Record host and CUDA-event intervals for preprocessing and complete inference separately.
CUDA-event intervals include host launch gaps. Do not call them isolated kernel latency.

In a separate pass, count 100 preprocessing calls per candidate with CuPy MemoryHook.
Record pool requests and requested bytes separately from underlying device allocations and device bytes.
A warm pool can serve requests without a new device allocation.
Require the fresh-output hook to observe at least 100 requests and 100 output payloads as a positive control.
Exclude hook instrumentation from timed measurement. It does not count TensorRT or Torch allocations.
The hook semantics were checked against [CuPy 14.2.0 documentation](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.cuda.MemoryHook.html).

## Acceptance

Require every checked frame to pass, complete raw sample coverage, and matching timing boundaries.
Recompute pooled and per-block p50/p95/p99 from raw rows.
A complete-host p99 regression above 5% on either fixture rejects adoption for that configuration.
A passing adoption guard means compatible performance, not proof of a material improvement.
Report the measured allocator reduction even if latency does not improve.
Do not infer lower whole-device memory or zero cudaMalloc from reduced pool requests.
The common preallocated slot prevents this comparison from estimating each strategy's independent peak memory.

## Reproduction and hardware

Generate controls and the engine from the same committed source and host before this comparison.
Run the existing six-fixture output-reuse correctness scenario first.

```bash
python scripts/yolo_reuse_benchmark.py --engine measured/yolo.engine --controls measured --output reuse-benchmark
python scripts/summarize_yolo_reuse.py reuse-benchmark --output reuse-verification.json
```

Use one independently terminated 30-minute GPU lease with export headroom.
Check live capacity and rates before provisioning. Reconcile the existing $15 budget.
Experiment 023 completed this protocol. All 40,000 frames passed.
Reuse removed pool requests but increased preprocessing host p50 by about 5.2%.
Complete-host p99 increased by about 0.2%, within the fixed guard. Keep reuse optional and fresh output as default.
See [the measured result](experiments/023-output-reuse.md).
