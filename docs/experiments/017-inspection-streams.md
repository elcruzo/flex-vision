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

## Initial result and revision

The first run at `0aca957` passed all 24 stream cases but failed the timing guard.
Odd-ROI pooled host p99 increased from 2.2092 to 2.5683 ms, approximately 16.3%.
The 4K gain remained. This failed observation must remain in the final report.

A revised implementation avoids constant event waits and constant `record_stream` calls on their original allocation stream.
PyTorch already orders and manages storage on that stream. Cross-stream constant tracking remains active.
Input storage tracking remains unconditional because its allocation stream is not inferred.
The extra host operations are a plausible overhead source, not a proven explanation for the tail spike.
Rerun correctness, stream checks, traces, and the complete timing protocol on the same lease.
Keep both measurements. Do not remove samples or replace the failed run.

## Revised result

Revision `406098256238803eb60a54cefea26f5b4561739c` passed all 24 cross-stream classifier cases.
The run used an L4, driver 580.159.04, CUDA 13.0, and PyTorch 2.9.1+cu130.
The largest tensor error was 0.0000631, below 0.0002. The largest logit error was 0.0000544.
Twelve retained outputs remained unchanged after later submissions. All class ordering and semantic checks passed.
Both revisions passed the 24-case stream test. The first failure concerned the timing guard, not numerical correctness.

Ordinary classifier validation, 38 CUDA edge checks, autocast rejection, and TF32 rejection also passed on the revised source.
Separate completed inference ranges recorded 199 kernels and zero memcpy events each for full and ROI strategies.
Those traces cover single-stream inference checks. The four-stream stress harness was not separately profiled.
Do not claim that its full execution was traced or that it measured multi-camera throughput.

The revised timing run used 7,200 samples and passed the predefined pooled guard.
Times below are milliseconds. Host totals exclude camera capture and input upload.

| Fixture | Implementation | Preprocess p50 / p99 | Host total p50 / p99 |
| --- | --- | ---: | ---: |
| Chelsea | Baseline | 0.1437 / 0.1808 | 2.1179 / 2.5341 |
| Chelsea | Revised ROI | 0.1528 / 0.1825 | 2.1376 / 2.2514 |
| Odd ROI | Baseline | 0.1442 / 0.1811 | 2.1120 / 2.5752 |
| Odd ROI | Revised ROI | 0.1546 / 0.1853 | 2.1512 / 2.4047 |
| 4K ROI | Baseline | 8.0212 / 8.1142 | 9.0653 / 9.1894 |
| 4K ROI | Revised ROI | 4.6608 / 4.6934 | 5.7039 / 5.7475 |

Revised 4K preprocessing p50 fell 41.89%, and p99 fell 42.16%.
Complete-host p50 fell 37.08%, and p99 fell 37.46%.
Across the four repeats, 4K baseline preprocessing medians ranged from 7.9861 to 8.0341 ms.
ROI medians ranged from 4.6539 to 4.6651 ms. The large-image benefit persisted in every repeat.

Small-image preprocessing remained 6–7% slower in the revised run.
Small-image host tails varied between runs, including the baseline's own tails.
The repeated result does not prove that the removed calls caused the initial p99 spike.
Preserve explicit strategy selection and investigate tail stability before making deployment guarantees.
The revised finite test supports stream handoff correctness. It does not establish bounded memory or concurrent throughput.

## Evidence and reproduction

Both runs retain all raw samples: [initial](data/017/initial/samples.csv) and [revised](data/017/revised/samples.csv).
Their [initial report](data/017/initial/latency.json) and [revised report](data/017/revised/latency.json) include pooled percentiles and allocator observations.
The [initial stream report](data/017/initial/streams.json) and [revised stream report](data/017/revised/streams.json) contain all case results.
Revised [full-frame](data/017/revised/full.json), [ROI](data/017/revised/roi.json), and [edge](data/017/revised/edges.json) reports record ordinary correctness.
See the [full trace summary](data/017/revised/planned-full-trace-summary.json) and [ROI trace summary](data/017/revised/planned-roi-trace-summary.json).

Each run exported 33 artifacts. All remote hashes matched local files before termination.
Both sets of 7,200 CSV samples independently reproduced the reported pooled percentiles.
The manifests retain original paths under ignored `benchmark-results/iteration-017-inspection/` and `benchmark-results/iteration-017b-inspection/`.
Large tensors and trace exports remain in those local directories. Small reports and raw samples are committed here.

```bash
python scripts/inspection_streams.py --output benchmark-results/inspection-streams
```

Use a clean revision and the pinned classifier weights. The runner requires CUDA.
For the latency protocol, use the commands in experiment 016 at the measured revision.

## Lease observations

CPU capacity changed during preflight. The Netherlands request was rejected before creating a resource.
A Romanian controller, `tkgx9yijdp4iuj`, was then created at 03:02 UTC on October 7.
The session paused longer than its 30-minute allowance. On resumption, no Pods or its temporary secret remained.
Its logs were unavailable, so its GPU creation and precise cleanup cause were not observed.
Reserve $0.35 for that attempt until billing reconciliation. Do not record it as a free failed startup.

The successful retry used California controller `s704oi3w6pj60p` and Romanian GPU `jm0qabsvc1bfg2`.
Its deadline was 04:11:43 UTC on October 7, or 9:11:43 p.m. Pacific on October 6.
Both measured revisions ran on that same GPU lease.
After artifact verification, the GPU was terminated early. Provider reads confirmed both Pods and both attempt secrets were absent.
Reserve $0.10 for the successful retry until its charges post. No persistent volume was created.
