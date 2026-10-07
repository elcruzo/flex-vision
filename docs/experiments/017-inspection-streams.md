# Experiment 017: inspection stream handoffs

## Contract and protocol

Preparation records an event after constant initialization.
Execution waits for that event and an optional caller-supplied input event.
The runtime records input and constant storage on its execution stream to prevent premature allocator reuse.
`submit` returns the output tensor and a completion event.
`result.wait()` enqueues a consumer dependency and records output storage on the consumer stream.
The wait does not block the host. The caller must enqueue inference on that consumer stream.
The producer must not overwrite input storage while preprocessing still reads it.

Use four distinct streams for preparation, frame production, preprocessing, and inference.
For each of three fixtures, run both strategies with four alternating original/mirrored frames.
Release input and handle references promptly. Delete each prepared object after its submissions.
Allocate replacement blocks on producer and processor streams to stress lifetime tracking.
Retain half the outputs and compare them after later submissions.
Check all 24 tensors, real classifier logits, top-five ordering, and cat-class expectations against independent CPU references.
Keep tensor tolerance 0.0002, logit tolerances 0.001 absolute and 0.0001 relative.
Reject missing CUDA instead of substituting a CPU test.

Repeat the integrated single-stream latency and trace checks after the runtime change.
Use the same four-repeat, 300-frame protocol and 20% 4K preprocessing improvement target.
Keep the maximum pooled host p99 regression at 5% on each fixture.
Do not infer concurrent throughput or hard memory bounds from this finite stress run.

The event and lifetime mechanisms follow [PyTorch events](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html)
and [record_stream](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.record_stream.html).

## Status

Prepared. CUDA stream and updated performance validation are pending.
