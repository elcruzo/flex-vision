# Sustained fixed YOLO protocol

## Outcome

Repeat the optimized detector comparison, then test complete inference under bounded offered load.
Check every accepted frame and allocator stability across repeated blocks.
This is the short developer tier. It does not replace the 30-minute soak or release tiers in TESTING.md.

The current detector runtime synchronizes each call. Use one persistent stream and four logical camera feeds.
Do not describe this harness as concurrent camera execution. It does not validate ROS transport or live capture.
Use the same resident 1080p person and cat inputs, fixed FP16 TensorRT engine, and CUDA NMS as experiment 020.
Both 1080p fixtures had exact tensors across all three candidates in that experiment.

## Frozen protocol

1. Repeat all six numerical cases and 60,000 matched timing samples from YOLO_VALIDATION.md.
2. Capture transfer evidence with its known control, separately from timing.
3. Reuse that engine for CPG, PyTorch, and two-pass CV-CUDA sustained blocks.
4. Warm each candidate for 100 fully checked frames before each block. Keep the stream and allocator pools across blocks.
5. Offer four staggered feeds at 60 FPS each for control load and 400 FPS each for overload.
6. Run each candidate for 30 seconds per load, with two repeats and rotated candidate order.
7. Give each feed two pending slots. Drop arrivals when its queue is full.
8. Dispatch the oldest admitted arrival across feeds. Record every offered frame, including dropped frames.
9. Drain pending work after arrivals stop. Abort if drain exceeds 30 seconds.
10. Export and independently check raw rows before terminating the rental.

The harness admits elapsed arrivals after each synchronous completion.
A delayed poll preserves their scheduled timestamps. It cannot observe arrivals during a blocked CUDA call.
Its bounded queues represent synthetic backlog, with at most eight pending frames and one frame executing.
They are not a transport implementation or a real-time scheduler guarantee.

Every accepted frame receives a GPU tensor, dense-output, validity, and final detection check.
Require exact FP16 tensor equality for these two known inputs on every candidate.
Use experiment 020's dense tolerances and final detection checks. No failed frame is discarded from the report.
Reference tensors and dense outputs must match the preceding numerical run before sustained measurement starts.
The same input through the same TensorRT engine defines the dense control.

The complete arrival boundary includes GPU validation and its scalar result transfer.
The pipeline event interval ends before correctness kernels. Validation still competes for resources and affects offered-load capacity.
No image payload is downloaded inside sustained execution. A scalar check flag is downloaded for each accepted frame.
These measurements cannot prove the uninstrumented application's sustained capacity.

## Acceptance and reporting

Require complete arrival coverage, no failed accepted-frame checks, and nonzero completion on every feed in every block.
Require pending queues to remain within their declared bounds.
Allow at most 1 MiB growth after drain in each declared allocator counter: Torch allocated/reserved and CuPy used/total bytes.
Report all four counters before and after each block and sample them every approximately 0.5 seconds.
Do not call these counters whole-device peak memory or TensorRT workspace measurements.
The TensorRT context stays alive throughout all blocks. Driver and external-library allocations remain outside these counters.

Recompute offered/completed/dropped counts, service rate, p50/p95/p99, and per-feed results from raw CSV files.
Report early and late p99 without inventing a new tail-drift threshold after measurement.
Zero drops at 240 offered FPS is a measurement outcome, not an assumed acceptance result.
Overload drops remain valid evidence. Report throughput improvements only if the matched samples support them.

The comparison success gate remains unchanged. Do not treat two photos as two product workloads.
Repeatability requires reporting the new complete-pipeline timing next to experiment 020, including regressions.
Keep the independent lease and existing $15 cumulative budget. Reconcile provider charges before provisioning.

## Commands

```bash
python scripts/yolo_sustained.py --engine benchmark-results/iteration-021-yolo/measured/yolo.engine \
  --controls benchmark-results/iteration-021-yolo/measured \
  --output benchmark-results/iteration-021-yolo/sustained
python scripts/summarize_yolo_load.py benchmark-results/iteration-021-yolo/sustained --output load-verification.json
```

Raw Nsight reports contain environment data. Keep them local.
Create a sanitized SQLite copy with `scripts/sanitize_trace.py` before sharing it.
Run the trace verifier again against the sanitized copy. Retain both original and sanitized hashes.


## Initial allocation finding

The first three control blocks completed without a correctness failure.
CV-CUDA's CuPy retained pool grew by one 2,457,600-byte FP16 output allocation. Live used bytes remained unchanged.
The declared 1 MiB retained-pool gate failed. Keep this result rather than calling the attempt stable.
The initial warmup omitted the per-frame validation path used during measurement.
The revised warmup checks all 100 frames through the same GPU validation and scalar completion path.
This tests a first-use allocation explanation without changing the growth allowance or clearing allocator caches.
The explanation remains a hypothesis until the revised GPU run completes.

## Saturation resource-lifetime investigation

The checked warmup passed all six control blocks but did not prevent CV-CUDA pool growth under overload.
The first overload CV-CUDA block added one output allocation at approximately 28 seconds, with no failed frame checks or live-byte growth.
Preserve this second failure. The warmup correction does not establish stable retained memory at saturation.

The pinned CV-CUDA source at `b051f32d3e17669c456d588103f820e78af35832` uses auxiliary-stream callbacks to release resource holds.
Its main `Stream.sync()` synchronizes only the submitted stream.
See [resource retention](https://github.com/CVCUDA/CV-CUDA/blob/b051f32d3e17669c456d588103f820e78af35832/python/mod_cvcuda/nvcv/Stream.cpp) and [wrapper caching](https://github.com/CVCUDA/CV-CUDA/blob/b051f32d3e17669c456d588103f820e78af35832/python/mod_cvcuda/nvcv/Tensor.cpp).
This provides a resource-lifetime hypothesis. It does not prove the source of the observed allocation by itself.

The next run collects all twelve normal blocks before rejecting a failed matrix.
No failed block becomes accepted. The verifier's explicit `--audit-failures` mode labels a complete failed matrix as `verified_failed_experiment`.
Missing arrivals, unchecked completed frames, inconsistent reports, and incomplete matrices still cause verification errors.

An optional final CV-CUDA overload diagnostic synchronizes its private auxiliary-stream hook after each preprocessing call.
It uses the same numerical checks, queues, duration, and memory allowance.
It does not clear allocator or vendor caches. The normal timed baseline remains unchanged.
This private version-specific hook is an investigation control, not a production requirement or a recommended vendor implementation.
Retain its result separately and exclude it from performance claims.
