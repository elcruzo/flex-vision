# Experiment 019 — Continuous inspection soak

Status: initial thermal gate failed. The retry is in progress.
No completed 30-minute result is claimed in this revision.

## Protocol

Run measured source `26ed35b` with `scripts/inspection_sustained.py --soak`.
Use four persistent camera streams, resident 4K BGR inputs, ROI preprocessing, and the pinned MobileNetV3 classifier.
Alternate the original and mirrored Chelsea fixture. Check every accepted tensor, logit vector, and ordered top-five result.
Each camera offers 60 frames per second. Each queue holds at most two pending frames.
Drop expired dispatch slots or arrivals to full queues. Preserve every offered frame in the raw CSV.

The [soak protocol](../SUSTAINED_INSPECTION.md#continuous-30-minute-soak) defines thermal stabilization and memory thresholds before measurement.
Condition under the same load for five minutes, with one additional five-minute interval if needed.
Then measure one uninterrupted 1,800-second arrival window, followed by a bounded drain.
This is a candidate stability test, not a new matched speed comparison.

## Initial attempt

GPU `mlrt00axvzkm05` in EUR-IS-2 ran the preflight and all 24 stream-handoff cases successfully.
The controller took approximately five minutes to start. A subsequent session delay consumed additional lease time before setup execution.
The independent deadline remained 2026-10-08 02:53:50 UTC.

Temperature reached approximately 88 degrees Celsius during the first conditioning interval.
The last 120 samples spanned 3 degrees Celsius. Their one-minute median change was 1 degree.
The fixed gate requires a range of at most 2 degrees, so it correctly rejected that interval.
The remaining deadline could not cover a second complete conditioning interval, the 30-minute soak, and verified export.
The second conditioning interval was interrupted. No measured soak began.

All ten completed artifact files matched the remote hashes before termination.
The [conditioning report](data/019/initial/sustained/conditioning-0.json), compressed raw CSV, and telemetry retain the failed gate evidence.
The remote manifest hashes the original CSV bytes. Decompress the committed CSV before comparing its hash.
Both Pods and temporary secret `txilc0n911` were absent after GPU termination.

## Retry

The retry uses the same source, workload, thresholds, and 60-minute independent lease.
Controller `gu2mqlrfefs9y0` uses EU-NL-1. Its requested GPU region is EUR-IS-1.
Reserve a further $0.65 within the original $15 authorization. Do not count this as new budget.
The retry expired during a session pause before SSH submission. No experiment ran.
Subsequent reads found no Pods and no temporary secret `jrdmqti3ir`.
The next attempt uses an automatic local coordinator for SSH submission, verified export, and termination.
Its cloud controller still enforces the independent deadline. The experiment source and thermal thresholds remain unchanged.
