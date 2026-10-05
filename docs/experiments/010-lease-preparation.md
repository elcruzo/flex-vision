# Experiment 010 — Independent lease and inspection harness preparation

Date: 2026-10-05. No GPU experiment ran in this iteration.

## Outcome

An independent lease workflow and a complete inspection baseline harness are committed.
The live shutdown test could not start because GitHub rejected the job before allocating a runner.
No workflow steps executed. A subsequent Runpod inventory contained no Pods.
The independent shutdown requirement remains open. There is no new CUDA or performance evidence.

The first queued run was canceled while correcting the API client and adding stale-dispatch rejection.
The corrected attempt was [workflow run 37369909079](https://github.com/elcruzo/flex-vision/actions/runs/37369909079).
Resolve runner availability before repeating the two-minute termination test.
Do not dispatch the 30-minute experiment until actual deadline deletion and provider absence are verified.

## Implemented preparation

- `e8cad83`: a GitHub-hosted lease creates one L4, records its exact ID, and deletes it at a fixed deadline.
- `e8639da`: CUDA classifier warmup plus raw inspection stage and complete-call latency samples.
- `dab9a45`: a named HTTP client and expiry for delayed workflow dispatch.

The lease starts its timer before container provisioning. It retains deletion authority from the creation response.
It checks the run marker and does not select resources by name.
The GPU container receives only the public SSH key and a run marker, not the Runpod API credential.
An encrypted GitHub Actions secret supplies the independent runner's credential.
See [the lease contract and failure limits](../GPU_LEASE.md).

The inspection benchmark requires successful CUDA classifier validation at the same clean source revision and environment.
It measures the existing fixed workload, including real MobileNetV3 Small inference.
The baseline still needs actual CUDA execution, trace export, and latency collection.
See [the inspection procedure](../INSPECTION_REFERENCE.md).

## Local checks

All 48 repository tests passed, including five cleanup fault checks.
Those checks cover identity mismatch, API retry and absence readback, cleanup after watcher failure, invalid duration, and overlapping rentals.
They do not establish live cloud termination.

The classifier CPU feedback loop passed all three fixtures after the warmup option was added.
That includes the 4K canvas and its ROI. CUDA warmup remains unexecuted.
Reports remain in the ignored `benchmark-results/inspection-warmup-cpu` directory.

A read-only Runpod API probe passed with the named HTTP client.
The default Python client received HTTP 403. No creation call was made by this probe.
The Mac probe used its existing system CA bundle. Certificate verification remained enabled.

## Cost and next acceptance

The billing read reports $5.007173080695793 across the five earlier rentals.
Experiment 008's final posted amount is $4.605089353630319, replacing its earlier $4.70 reserve.
This iteration created no billable Runpod resources.
The remaining total authorization is approximately $9.99 before a later lease.

Next evidence must show:

1. A hosted runner creates and terminates a disposable Pod at the two-minute deadline.
2. Runpod confirms the exact Pod is absent after termination.
3. A separate bounded lease runs the CUDA classifier validation and absent-class negative control.
4. A separate trace records transfers and launches within completed inference ranges.
5. Raw repeated samples establish the unfused second-workload baseline.
6. Exported artifact hashes match before normal Pod termination.

These steps do not by themselves pass the two-workload optimization gate.
