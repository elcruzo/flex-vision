# Stream ownership and asynchronous execution

Status: proposed asynchronous contract. The public detector remains synchronous.
Experiment 023 does not support selecting caller-owned output for faster execution.
This document defines the next hypothesis: explicit dependencies can remove host waits without corrupting outputs or increasing tails.

## Current baseline

CuPy callers order production on the preprocessing stream or provide an explicit producer-event wait.
DLPack imports establish their protocol dependency. They do not make later input mutation safe.
The caller retains the input and does not modify it until preprocessing completes.
CPG synchronizes its preprocessing stream before returning the output.
The fixed TensorRT harness also synchronizes its consumer stream before returning.
These waits prevent this baseline from demonstrating overlap.

`scripts/yolo_stream_ownership.py` prepares 48 real-inference cases across six fixtures and two input frameworks.
Each case uses distinct persistent producer, preprocessing, and consumer streams.
It checks explicit producer handoff, exact tensors, dense outputs, final detections, and retained default outputs.
This harness is prepared, not GPU-validated. It measures no overlap or performance benefit.

## Proposed submission contract

An asynchronous submission must retain the original input, imported view, destination, preprocessing stream, and readiness event.
A returned completion object must distinguish submission from GPU completion.
Waiting on its readiness event orders consumer work. It does not release the input before preprocessing completes.
The caller must not mutate source storage while preprocessing reads it.
The output owner must survive every consumer read, including work on another stream.
DLPack transport does not replace that lifetime requirement.

Use a separate explicit submission API. Keep `pipeline(frame)` synchronous until acceptance supports a default change.
The proposed submission API and completion object are not implemented.
Do not return an ordinary array with an undocumented pending write.

## Reusable slot states

A managed slot moves through free, preprocessing, ready, consumer-active, and recyclable states.
Associate each reservation with a generation to reject stale completion notifications.
A slot becomes recyclable only after every registered consumer completion event finishes.
An unregistered consumer keeps reuse unsafe. CPG cannot infer external readers from a pointer.
Never recycle a slot solely because preprocessing finished or a Python reference disappeared.

Start with one explicit consumer per submission and bounded slots.
Reject unsupported devices and streams before launch.
Do not use garbage collection as the primary completion mechanism.
On a failed launch or shutdown, drain outstanding work before releasing owners.
A device failure must remain visible rather than returning the slot as safely free.
Multiple consumers and concurrent host threads need separate acceptance before support claims.

## Acceptance before selection

1. Run the prepared three-stream baseline through the fixed TensorRT consumer.
2. Add an asynchronous consumer harness that retains its execution context and output until its completion event.
3. Delay producer and consumer work and verify ordering without host synchronization in the submission path.
4. Exercise alternating slots, retained outputs, source-owner release, shutdown, and stale generations through real inference.
5. Compare identical fresh synchronous and asynchronous strategies with declared queue and timing boundaries.
6. Capture the absence of unwanted host waits and pixel transfers with Nsight Systems.
7. Require complete numerical coverage, bounded memory, and a predeclared end-to-end p99 regression guard.

Missing a handoff is a caller error. Do not introduce a nondeterministic GPU race as the sole negative control.
Test rejected state transitions deterministically and validate correct event handoffs on hardware.
Separate correctness instrumentation from timing, while recording its effect on later samples.
Do not use local tests or skipped GPU jobs as acceptance for these steps.

## API evidence

The stream APIs support event ordering through [CuPy Stream](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.cuda.Stream.html).
Allocator lifetime and stream ordering are distinct concerns. Review [PyTorch record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html) before Torch-owned async storage.
Use the installed pinned versions when selecting the implementation.
Raw TensorRT bindings require explicit owner retention. A pointer binding does not establish a Python owner.

## Guarded experiment entry point

`scripts/run_yolo_stream_experiment.sh` regenerates the engine, numerical controls, baseline timings, and transfer capture on the experiment host.
It then runs the 48 stream executions and checks report coverage with `scripts/verify_yolo_stream_ownership.py`.
The report records three distinct stream handles, every cycle's checks, and shutdown completion.
The verifier rejects missing cases, duplicates, unchecked cycles, shared handles, dirty source, and failed stream drainage.
This report audit does not independently recompute tensors. Retain the full numerical controls and use the existing numerical verifier separately.

The entry point requires the independent lease deadline and leaves five minutes for export.
Use the existing direct controller and reconcile delayed charges before dispatch.
No new rental ran for this preparation. GPU acceptance remains pending.

Local verification passed eight evidence fault checks, Python compilation, and shell syntax.
Those checks validate the harness plumbing only. They do not establish event ordering on NVIDIA hardware.
