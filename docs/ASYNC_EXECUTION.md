# Stream ownership and asynchronous execution

Status: experimental owned-output submission passed the fixed 48-execution TensorRT scenario.
Broader async acceptance and performance remain pending.
The default detector call remains synchronous.
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
Experiment 024 passed all 48 executions on RTX 4090. It measures no overlap or performance benefit.

## Proposed submission contract

An asynchronous submission must retain the original input, imported view, destination, preprocessing stream, and readiness event.
A returned completion object must distinguish submission from GPU completion.
Waiting on its readiness event orders consumer work. It does not release the input before preprocessing completes.
The caller must not mutate source storage while preprocessing reads it.
The output owner must survive every consumer read, including work on another stream.
DLPack transport does not replace that lifetime requirement.

Use a separate explicit submission API. Keep `pipeline(frame)` synchronous until acceptance supports a default change.
`Pipeline.submit(frame, stream=...)` now implements an experimental completion object for fresh owned outputs.
It requires an explicit CuPy stream on the current device. Default stream singletons are rejected.
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
[Experiment 024](experiments/024-detector-streams.md) completed the fixed synchronous scenario. Asynchronous acceptance remains pending.

Local verification passed eight evidence fault checks, Python compilation, and shell syntax.
Those checks validate the harness plumbing only. They do not establish event ordering on NVIDIA hardware.

## Experimental submission API

```python
with pipeline.submit(frame, stream=preprocess_stream) as pending:
    tensor = pending.wait_on(consumer_stream)
    # Enqueue the consumer on consumer_stream and retain tensor until it finishes.
    # The context exit waits for preprocessing, not consumer completion.
```

`wait()` establishes host completion and returns the owned output.
`wait_on(stream)` queues a consumer dependency without a host wait. It does not release source owners.
`close()` waits for preprocessing and releases source owners only after successful completion.
The completion object retains output and stream owners. It does not track external consumers.
The caller retains the returned tensor through consumer completion and does not mutate pending input.
Caller-owned async output, slot recycling, concurrent host access, graph capture, and consumer cancellation are unsupported.
Use a native CuPy stream, or a stream wrapper that retains its foreign owner.
A raw ExternalStream pointer does not retain the foreign stream owner automatically.

Use explicit close or a context manager. Object destruction has a conservative blocking fallback.
Do not rely on garbage collection for normal completion or error reporting.
A dispatch error drains the stream before local input owners are released.
A completion error keeps source owners retained and propagates to the caller.
Device failure recovery is not established by this experimental path.

Three local lifecycle checks verify retention across handoff, completion failure, and wrong-device rejection.
They use fake events and streams. Experiment 025 adds fixed CUDA/TensorRT correctness evidence.
Absence of host waits, delayed-consumer behavior, and performance remain unmeasured.
The next hardware harness must exercise this implementation through delayed producer and consumer work.

The guarded async correctness entry point is scripts/run_yolo_async_experiment.sh.
It tests 48 owned-output submissions and releases caller source references before TensorRT consumption.
The TensorRT consumer remains synchronous. This scenario does not measure overlap or performance.
Require the async-submit report mode and source-release checks when auditing this scenario.

[Experiment 025](experiments/025-async-submission.md) passed all 48 owned-output submissions after a pinned stream compatibility correction.
The consumer still synchronizes. Keep the default unchanged and test delayed consumers before performance selection.

## Prepared delayed-consumer gate

`scripts/yolo_delayed_consumer.py` prepares 48 real-inference ownership checks.
The separate PendingConsumer permits one outstanding TensorRT enqueue and retains its input, output, context, and native stream owner.
It rejects another enqueue until completion succeeds. A failed completion keeps owners and the busy context retained.
The synchronous baseline consumer is unchanged. This new harness consumer remains GPU-unvalidated.

Warm the context before introducing a finite 500,000,000-cycle single-thread delay on the consumer stream.
This delay uses CUDA clock64 and does not define a wall-clock duration or benchmark metric.
After enqueue, close preprocessing and delete caller output references.
Require the consumer event to remain incomplete at that observation. Reject a run where it already completed.
Then reject context rebinding, allocate another preprocessing result, and check both tensors and real detector results.
The pending consumer retains the original tensor until its own completion fence.
Drain both streams and any outstanding jobs before shutdown.

`scripts/run_yolo_delayed_experiment.sh` connects the scenario to the existing independent lease workflow.
`scripts/verify_yolo_delayed_consumer.py` rejects missing cycles, completed readers, duplicate fixtures, and failed drainage.
Local lifecycle and evidence checks passed. GPU acceptance, delayed-context behavior, and performance remain pending.
No new rental ran for this preparation. Reconcile the existing budget before hardware execution.

API references: [TensorRT Python runtime](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html) and
[CUDA 13.0 clock64](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#time-function).

Experiment 026 missed export during a local session pause. No delayed-consumer GPU acceptance is available.
See [the outcome and coordinator correction](experiments/026-expired-delayed-consumer.md).

The [matched serial comparison](ASYNC_BENCHMARK.md) now has a prepared TensorRT/NMS runner and independent sample audit.
It requires passing delayed-consumer evidence from the same source and engine before collecting timing.
This preparation establishes no hardware performance result and does not change the synchronous default.

The delayed-consumer entry point now regenerates all six numerical fixtures with `yolo_gpu.py --validate-only`.
It builds the same engine and checks every existing candidate, framework import, negative stride, and retained output.
It omits repeated baseline latency sampling and baseline trace installation before the pending-reader test.
This is a correctness dispatch, not new comparison or trace evidence. The numerical policy remains unchanged.
The matched comparison and async-specific trace still need separate execution and acceptance.
