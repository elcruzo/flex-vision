# Experiment 021 — Detector repeatability and bounded load

Status: the first repeated comparison passed. The first sustained attempt failed its retained-pool growth guard.
The revised checked-warmup GPU execution remains pending.

## Repeated comparison

Run source `6d396a3` with the same fixed detector backend and protocol as experiment 020.
All six numerical cases, 18 independent NumPy checks, and 60,000 matched samples passed again on an RTX 4090.
The transfer verifier passed on a sanitized SQLite export, including the known 4,096-byte control.
The original capture hash remains in the remote manifest. Raw environment-bearing captures stay local.

| Fixture | Candidate | Preprocess p50 / p99, ms | Complete host p50 / p99, ms |
| --- | --- | --- | --- |
| Person 1080p | CPG | 0.0850 / 0.0965 | 0.8749 / 0.9257 |
| Person 1080p | PyTorch | 0.1023 / 0.1290 | 0.8990 / 0.9852 |
| Person 1080p | CV-CUDA | 0.1158 / 0.1423 | 0.9155 / 0.9919 |
| Cat 1080p | CPG | 0.0850 / 0.0994 | 0.8742 / 1.0087 |
| Cat 1080p | PyTorch | 0.1004 / 0.1432 | 0.8918 / 1.0597 |
| Cat 1080p | CV-CUDA | 0.1147 / 0.1601 | 0.9112 / 1.0839 |

CPG preprocessing p50 improved by 25.92–26.59% versus CV-CUDA, compared with 24.77% in the preceding optimized rental.
Its preprocessing p99 improved by 25.22–30.58% versus PyTorch. The preceding rental measured 37.10–38.66%.
Complete host p50 improved by 1.97–2.68% versus PyTorch and 4.07–4.44% versus CV-CUDA.
Both rentals support the scoped preprocessing improvement. Their differing tails require continued reporting rather than a universal latency claim.
The two photos remain one detector workload. They are not two independent product workloads.

## Initial bounded-load result

The [sustained protocol](../SUSTAINED_YOLO.md) uses four logical feeds and one persistent serial stream.
The first three 30-second control blocks offered 7,200 frames each at an aggregate 240 FPS.
CPG completed 7,191 frames and dropped nine. PyTorch and CV-CUDA completed all 7,200.
All 21,591 completed frames passed their GPU tensor, dense-output, and detection checks.
The partial CSV audit verified arrival coverage and completed-frame checks. It does not pass full sustained acceptance.

Torch allocated/reserved and CuPy live used counters showed zero post-drain growth in these three blocks.
CPG and PyTorch also showed zero retained CuPy growth.
CV-CUDA's retained CuPy pool grew by 2,457,600 bytes, exactly one FP16 1×3×640×640 output allocation.
The declared allowance is 1,048,576 bytes. This failed the guard and stopped the experiment before overload and second-repeat coverage.
No sustained throughput advantage is claimed from this incomplete attempt.

The initial warmup omitted the validation used on every measured frame.
Source `6b0af27` changes warmup to 100 fully checked frames on each candidate.
It preserves the stream, allocation pools, admission policy, and growth threshold. No cache is cleared.
This tests a first-use retained-allocation explanation. It does not retroactively pass the initial attempt.

## Evidence and cleanup

The [initial evidence](data/021/initial/results.json) includes raw compressed comparison and load rows, independent verifier results, and sanitized trace data.
All 72 remote artifacts matched their hashes before termination at 05:50:04 UTC on October 9, 2026.
The controller and temporary credential were absent at 05:50:15 UTC.
The retry uses a new independently bounded lease within the original $15 authorization.

This experiment does not complete detector soak, concurrent streams, batching, reusable workspace, live-camera, Jetson, or ROS acceptance.
