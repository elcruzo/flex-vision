# Experiment 026: delayed-consumer dispatch lost during local pause

## Outcome

No delayed-consumer result bundle was exported. The scenario remains unvalidated.
Do not infer a numerical pass or failure from the expired rental.
The submitted source was `234d053`, with the prepared guarded delayed-consumer entry point.
The last available progress showed fixed model export warnings and SSH polling failures.
No tensor, detector, pending-reader, or shutdown acceptance is established from that progress.

## Session pause and cleanup

Controller `yx40212i3ogfw4` was created at 00:20:49 UTC on October 10, 2026.
GPU `ck71jv7pv25pug` received the experiment at 00:21:26 UTC.
The independent lease was thirty minutes. The local coordinator ran under caffeinate to inhibit idle sleep.
The local session nevertheless paused across the lease deadline. Idle-sleep inhibition does not guarantee session continuity.
Subsequent SSH retries continued after the cloud expiration because local monotonic elapsed time lagged wall time.

A read at 02:19:25 UTC found no Pods and confirmed the GPU, controller, and temporary secret `bp5fp6tahn` absent.
The coordinator then received a local interrupt and reconfirmed exact resource absence in its cleanup path.
No termination of unrelated resources occurred. No rental remains active.
The first absence read does not establish an exact earlier deletion timestamp.
Provider compute charges for this attempt were approximately $0.4791 including disk at the billing read.
Those charges are consistent with the guarded lease, but billing alone is not a deletion-event log.
See [cleanup evidence](data/026/cleanup.json).

## Coordinator correction

The coordinator now compares wall time against the independent deadline minus five minutes before every progress poll.
It also checks the actual deadline again before export.
A resumed session beyond its export allowance raises promptly and runs exact cleanup instead of continuing SSH retries.
A regression test freezes monotonic time while advancing wall time beyond the cloud deadline.
It verifies no post-expiration SSH progress poll and exact GPU cleanup.
All four coordinator fault checks passed locally. This correction does not recover the expired artifacts.

Independent cleanup remains useful when the Mac pauses. Artifact export still needs an awake, connected coordinator.
A durable export path independent of the Mac is a separate infrastructure improvement, not implemented support.
Do not claim that caffeinate guarantees export or extend the GPU lease silently to compensate for pauses.

## Budget reconciliation

Posted Pod billing across the requested window is $9.770127805024458.
Newly posted charges for both experiment 025 attempts and experiment 026 total $0.8992737082371605.
Those resources already had reservations, so offset that posted amount against their remaining reservation headroom.
Keep the full $0.65 reservation ceiling for experiment 026 and previous delayed-charge headroom.
The conservative cumulative bound is $14.780820900171262, within the authorized $15.
This bound is not actual spending. Reconcile existing ceilings before any further paid retry.

## Next gate

Run the delayed-consumer scenario with a reliable export window or implement durable independent result export first.
Keep its required pending-reader observation, exact tensors, model results, and shutdown checks unchanged.
The explicit async submission still has experiment 025's fixed correctness evidence.
Delayed consumers, async-specific traces, and matched latency remain open. No product milestone is closed by this attempt.
