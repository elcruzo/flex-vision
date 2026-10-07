# Experiment 016: integrated inspection CUDA acceptance

## Protocol

Validate `planned-full` and `planned-roi` through the real CUDA classifier at the same clean revision as the independent baseline.
Retain the existing tensor, logit, top-five, and semantic tolerances.
Compare the integrated ROI path with the baseline using experiment 014's alternating four-repeat, 300-frame protocol.
Require at least 20% lower 4K preprocessing p50 or p99 and no pooled host p99 regression above 5% on any fixture.
Capture separate traces for both integrated strategies. Do not use profiled latency for the comparison.
Check CUDA edge crops, retained outputs, and precision rejection through the public API.
Record all results, including failed criteria. No automatic strategy threshold follows from three fixture sizes.

## Result

The integrated API passed this bounded CUDA acceptance run on October 6, 2026, Pacific time.
The measured clean revision was `b77e90c`. The host used an L4, driver 580.159.04, CUDA 13.0, and PyTorch 2.9.1+cu130.
TorchVision was 0.24.1+cu130. TF32 and cuDNN benchmark selection were disabled.
See the [package manifest](data/016/packages.txt).

All nine ordinary classifier cases passed: three each for baseline, integrated full-frame, and integrated ROI execution.
All six separately profiled integrated cases also passed.
The maximum tensor error was 0.0000461, below the unchanged 0.0002 limit.
Logit comparisons, ordered top-five labels, and cat-class expectations passed.
Another 38 CUDA edge cases passed with maximum tensor error 0.00000382.
Retained outputs and input data remained unchanged. Autocast and cuDNN TF32 rejection checks passed.

## Matched timing

Times are milliseconds. Each implementation and fixture has four repeats of 300 samples.
The host interval ends after CUDA completion. Camera acquisition and upload are excluded.

| Fixture | Implementation | Preprocess p50 | Preprocess p99 | Host total p50 | Host total p99 |
| --- | --- | ---: | ---: | ---: | ---: |
| Chelsea | Baseline | 0.1485 | 0.1875 | 2.2463 | 2.5142 |
| Chelsea | Integrated ROI | 0.1517 | 0.1898 | 2.2425 | 2.5599 |
| Odd ROI | Baseline | 0.1460 | 0.1833 | 2.1956 | 2.4762 |
| Odd ROI | Integrated ROI | 0.1495 | 0.1843 | 2.1802 | 2.4302 |
| 4K ROI | Baseline | 7.9890 | 8.0915 | 9.0299 | 9.1673 |
| 4K ROI | Integrated ROI | 4.6404 | 4.6882 | 5.6815 | 5.7498 |

The 4K preprocessing p50 fell 41.91%, and p99 fell 42.06%.
Complete-host p50 fell 37.08%, and p99 fell 37.28%.
The largest complete-host p99 regression was 1.82% on Chelsea, below the predefined 5% limit.
Small-image preprocessing medians were 2.2–2.4% slower. Explicit strategy selection remains appropriate.

| Repeat | 4K baseline preprocessing p50 / p99 | Integrated ROI preprocessing p50 / p99 |
| --- | ---: | ---: |
| 0 | 7.9420 / 8.1037 | 4.6303 / 4.6883 |
| 1 | 7.9872 / 8.0609 | 4.6312 / 4.6891 |
| 2 | 7.9919 / 8.0683 | 4.6445 / 4.6842 |
| 3 | 8.0063 / 8.0873 | 4.6464 / 4.6870 |

Peak PyTorch allocation above the starting allocation was 399,261,696 bytes for the baseline and 176,454,144 bytes for integrated ROI execution.
Those observations cover the measured path. They are not total GPU memory or a guaranteed workspace bound.

## Transfer evidence

Both integrated strategies recorded 199 kernels in each of three completed `inspection_to_classifier` ranges.
All six ranges recorded zero memcpy events and bytes.
Fixture uploads, preparation, warmup, and validation downloads occurred outside the ranges.
The range includes classifier inference and completion synchronization.
This establishes residency for these synchronous test boundaries, not arbitrary cross-stream or ROS transport behavior.

## Reproducible evidence

The [raw CSV](data/016/samples.csv) contains 7,200 samples.
Independent local recomputation matched all pooled percentiles in the [latency report](data/016/latency.json).
The report includes p95, inference intervals, allocator observations, source hashes, and validation hashes.
See the [baseline](data/016/baseline.json), [integrated full-frame](data/016/full.json), and [integrated ROI](data/016/roi.json) correctness reports.
The [edge report](data/016/edges.json) records additional contract checks.
Separate [full-frame](data/016/planned-full-trace-summary.json) and [ROI](data/016/planned-roi-trace-summary.json) summaries record kernels and copies.

All 32 exported artifacts matched the [remote SHA256 manifest](data/016/sha256.txt) before GPU termination.
Large arrays, SQLite exports, and Nsight reports remain locally under ignored `benchmark-results/iteration-016-inspection/`.
The manifest retains those original paths. The small reports and samples are committed here.

```bash
python scripts/local_classifier.py --device cuda --output benchmark-results/baseline
python scripts/local_classifier.py --device cuda --implementation planned-full --output benchmark-results/full
python scripts/local_classifier.py --device cuda --implementation planned-roi --output benchmark-results/roi
python scripts/inspection_benchmark.py --validation benchmark-results/baseline/report.json --candidate-validation benchmark-results/roi/report.json --output benchmark-results/comparison
```

Use new output directories, pinned weights, and a clean revision. The benchmark checks implementation identity and runtime source hashes.

## Scope and cleanup

The fixed public inspection API now has CUDA correctness, residency, and matched ROI timing evidence on this L4.
Automatic selection, general inspection graphs, cross-stream acceptance, representative industrial data, Jetson, and ROS remain pending.
The broader release gate remains open. The local CPU/MPS checks and this bounded GPU run do not establish all-platform support.

GPU `n2x2oog9mcnytl` used controller `nvah9ciyl9tjqq` at $0.49/hour and $0.06/hour, excluding disk.
The controller armed a deadline of 03:21:03 UTC on October 7, or 8:21:03 p.m. Pacific on October 6.
After export verification, the GPU was terminated early.
At 02:57 UTC, provider reads confirmed both Pods and temporary secret `thh9dns536` were absent.
Reserve $0.10 for this run until provider billing posts.
