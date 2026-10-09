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

## Completed short matrix

The final GPU run passed all twelve normal blocks and a separate auxiliary-drain diagnostic.
The independent verifier checked 331,200 offered records and 187,279 correct completed frames.
All declared post-drain allocator growth counters were zero. Earlier failures remain failed.
Normal CV-CUDA also had zero growth in this rental, so the auxiliary diagnostic does not isolate the earlier cause.
See [experiment 021](experiments/021-detector-repeatability.md) for variable throughput, tails, drops, and the next soak gate.

## Continuous detector soak: prepared, not measured

Use `--soak` to run CPG through the same fixed detector and GPU correctness checks.
This mode replaces the short comparative matrix. It does not run comparative vendor soak blocks.
Keep the TensorRT context, resident fixtures, stream, and allocator pools alive throughout conditioning and measurement.

1. Run five minutes at four logical feeds of 60 FPS each.
2. Require the last 120 telemetry samples to span at most 2°C.
3. Require the two consecutive 60-sample median temperatures to differ by at most 1°C.
4. Allow one additional five-minute conditioning attempt if temperature does not stabilize.
5. Run one continuous 1,800-second arrival schedule with two pending slots per feed.
6. Check every completed frame through preprocessing, TensorRT, CUDA NMS, and GPU correctness checks.
7. Retain every offered arrival, including queue drops, and collect allocator samples about twice per second.
8. Retain GPU temperature, clocks, power, utilization, and device memory once per second.
9. Independently audit the retained rows and telemetry before accepting the result.

The measured schedule offers 432,000 frames. Completion count and drops remain measured outcomes.
Do not reset queues, allocator pools, or the context during the measured interval.
The final drain may extend execution by at most 30 seconds.
Require no failed completed-frame checks and nonzero completion on all four feeds.
Keep the 1 MiB post-drain allowance for all four declared allocator counters.
Also compare sampled counter ceilings during the first and last five minutes against that allowance.
Report six five-minute latency distributions without imposing a new tail-drift threshold after measurement.
Require telemetry continuity and complete allocator sampling. Missing telemetry fails acceptance.
These GPU counters do not measure Jetson power or prove thermal behavior on Jetson.

```bash
python scripts/yolo_sustained.py --soak \
  --engine benchmark-results/iteration-022-yolo/measured/yolo.engine \
  --controls benchmark-results/iteration-022-yolo/measured \
  --output benchmark-results/iteration-022-yolo/soak
python scripts/summarize_yolo_soak.py benchmark-results/iteration-022-yolo/soak \
  --output soak-verification.json
```

The script retains host dictionaries for all arrivals until CSV export.
The fixed control load bounds this at 432,000 rows. Record host memory during the first GPU run.
Do not increase this mode to overload or extend its duration without reviewing host-memory requirements.
The local fault checks validate the evidence verifier. They do not validate CUDA execution or complete the soak gate.

A 60-minute independently terminated lease must cover setup, model and engine preparation, up to ten minutes of conditioning, and thirty minutes of measurement.
Preserve five minutes for export. Do not launch if the prepared setup cannot meet that bound.
The current $15 budget includes reservations for unposted charges. Reconcile those charges before reserving another lease.
No detector soak rental was started for this preparation.
