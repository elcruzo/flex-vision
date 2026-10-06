# Experiment 013: restrict inspection work to the required region

## Hypothesis and semantics

Experiment 012 measured approximately 8 ms of preprocessing for the 4K inspection fixture.
The candidate computes only the requested crop and its filter neighborhood.
It retains Gaussian, sharpen, clamp, resize, and normalization as separate operations.

Gaussian radius two and sharpen radius one require a three-pixel halo around the crop.
The candidate clips this extended rectangle to the original image bounds.
Replication at those bounds therefore retains the original full-frame border semantics.
Artificial work-rectangle borders remain outside the final crop's dependency region.
This argument concerns pixel dependencies. Backend arithmetic still requires numerical checks on each device.

For the current 4K ROI, the work rectangle is 2712 × 1806 instead of 3840 × 2160.
That reduces the number of pixels entering conversion and filters by approximately 41%.
This is a geometric count, not a measured latency improvement.

## Local evidence

Both CPU and MPS preprocessing passed all three photo cases through real CPU MobileNetV3 inference.
The existing tensor, logit, ordered top-five, and cat-class checks retained their original tolerances.
An additional 38 CPU/MPS cases checked small images, corners, edges, and single-pixel crops against independent NumPy results.
Their maximum tensor error was `0.00000430`, below `0.0002`.
The repository checks passed: 54 tests, including retained-output and edge checks.

Reproduce the full classifier checks with a new output directory:

```bash
.venv/bin/python scripts/local_classifier.py --device cpu --implementation roi --output benchmark-results/iteration-013-roi-cpu
.venv/bin/python scripts/local_classifier.py --device mps --implementation roi --output benchmark-results/iteration-013-roi-mps
```

The reports record candidate identity and its source hash.
The baseline benchmark rejects validation from the ROI candidate, preventing an implementation mismatch.

## Next acceptance step

Run baseline and candidate validation at the same clean revision on CUDA.
Then collect alternating repeated measurements through the same classifier, with identical resident input, precision, warmups, and synchronization.
Preserve raw samples and repeat-level distributions. Capture transfer and launch evidence in a separate profiling run.
Require unchanged numerical tolerances and inspect p99 as well as median latency.

No paid rental was created for this local step.
CUDA correctness and candidate speed remain unmeasured.
The local MPS runner downloads results for CPU inference, so this step does not establish GPU residency.
This candidate remains experimental and does not extend the public CPG operator catalog.
