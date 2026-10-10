# CUDA Array Interface real-inference gate

Status: experiment 031 passed the fixed 144-case correctness protocol on RTX 4090. Broader acceptance remains pending.
This gate advances R05 interoperability and R10 ownership without expanding the filter catalog.

## Fixed scenario

Use the pinned YOLO FP16 TensorRT engine and six independently checked numerical controls.
Require a clean committed source and record the engine hash.
The exporter exposes only CUDA Array Interface version 3, so DLPack cannot bypass the new adapter.
It retains the backing allocation and advertised native producer stream.

The matrix has 144 executions:

- Six existing photograph, padded, and 1080p fixtures.
- Contiguous, reversed row, and reversed channel views with the same logical input pixels.
- Explicit, legacy default, per-thread default, and already-ready producer contracts.
- Synchronous calls and owned-output submissions.

Initialize storage separately, then queue a finite 500,000,000-cycle producer delay before copying the correct pixels.
Require the producer event to remain pending before import.
For a stream=None export, complete this event before importing because that contract declares readiness.

For submit, queue a 1,000,000,000-cycle preprocessing delay first.
Require preprocessing completion to remain pending when submission returns.
Release the caller's exporter reference and require its weak reference to remain alive before explicit completion.
For an advertised stream, require pending completion again immediately before queuing a zero fill on that stream.
The adapter's reverse dependency must keep that write behind preprocessing's input read.
For stream=None, the caller must wait for preprocessing completion before queuing any future write.
The adapter cannot order writes on a stream that the exporter does not advertise.
Protocol cai-v2 records these distinct reuse contracts and the observation at the actual reuse boundary.
This deliberately pending read prevents a missing dependency from passing merely because preprocessing already finished.

Wait for preprocessing, then check every output bit against the independent oracle.
Run actual TensorRT inference and compare dense outputs and detections with the same-engine controls.
Check the preceding retained output after each subsequent call.
Finally drain the producer and confirm its requested zero fill completed.
Report absent retention observations as null for the first output and synchronous exporter-lifetime boundary.
Do not invent those observations as successful checks.

The local verifier rejects missing or duplicate scenarios, missing pending boundaries, failed retention, missing source identity, and incomplete inference.
It verifies report coverage, not independent numerical recomputation.
Retain failed reports and drain all known streams before writing shutdown status.

## Execution

Use `scripts/run_yolo_cai_experiment.sh` inside the independently bounded GPU session.
It regenerates numerical controls first and reserves five minutes before the cloud deadline for export.
The explicit hardware command is:

```bash
.venv-gpu/bin/python scripts/yolo_cai.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" \
  --output "$CPG_RESULTS/cai.json"
.venv-gpu/bin/python scripts/verify_yolo_cai.py "$CPG_RESULTS/cai.json" \
  --output "$CPG_RESULTS/cai-audit.json"
```

Reconcile the existing $15 budget before another paid rental.
Unit-test passes do not complete this scenario.
No paid resource is created by preparing this harness.

## Remaining acceptance

This is a correctness experiment with artificial delays. It provides no latency or overlap claim.
A dedicated transfer capture remains required for CAI residency acceptance.
Read-only, zero-stride, invalid-device, invalid-pointer, external-library, and broader dtype/platform coverage remain pending.
Invalid pointer tests must avoid corrupting the main measured CUDA context.
An exported stream handle still relies on the producer retaining its actual native owner.
The runtime cannot reconstruct ownership from an integer alone.


The first hardware attempt failed before CAI import because it used Torch's event query spelling on CuPy.
The corrected harness uses documented CuPy Event.done at both pending boundaries.
See experiments/029-cai-event-failure.md. Require a fresh complete result before acceptance.

Experiment 030 passed seven scenarios before unordered stream=None reuse failed in the harness.
See experiments/030-cai-ready-reuse.md. Corrected cai-v2 GPU acceptance remains pending.

Experiment 031 passed cai-v2 after the preserved failures in 029 and 030.
See experiments/031-cai-inference.md for exact scope. Transfer capture remains pending.
