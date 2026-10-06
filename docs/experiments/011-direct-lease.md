# Experiment 011 — Direct Runpod lease

Date: 2026-10-05, America/Los_Angeles. Event timestamps below use UTC.
Source revision: `79a6db7`.

## Outcome

Independent cleanup passed without GitHub Actions or a Mac-side termination command.
A disposable CPU controller created one L4 GPU Pod, armed its deadline, and later removed both Pods and its temporary secret.
Provider reads confirmed that no Pods remained and that the exact temporary secret was absent.
GitHub Actions is now optional. Its billing or runner availability does not block direct GPU experiments.

This was a shutdown test only. It did not run CUDA preprocessing, model inference, or a latency benchmark.
The second-workload GPU baseline remains pending.

## Resources and observations

| Resource | Identity | Configuration |
| --- | --- | --- |
| CPU controller | `kw32niruy1iajz` | `cpu3c`, two vCPUs, EU-RO-1, two GB container disk |
| GPU | `w8mv4bvnfx3zaw` | One NVIDIA L4, EU-RO-1, 50 GB container disk |
| Temporary encrypted secret | `gqdgjykv2s` | Credential available only to the CPU controller |
| Lease identifier | `1791249848-803885082383` | Used with exact returned IDs for identity checks |

The CPU image was `python:3.12-slim-bookworm` from the official Python image project.
The GPU image was `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`, read from Runpod template `a9dk3g7cny`.
No persistent or network volume was created. The CPU controller exposed no ports. The GPU exposed only SSH.

Observed events:

- 01:24:09.848 UTC: provider creation time for the CPU controller.
- 01:24:16.796 UTC: controller log reported `controller_ready`.
- 01:24:17.844 UTC: controller started GPU creation with deadline `1791249977.8440638` (01:26:17.844 UTC).
- 01:24:18.368 UTC: provider creation time for the GPU.
- 01:24:19.673 UTC: controller log reported `armed` with the exact GPU ID and deadline.
- By 01:29:41 UTC: the controller log endpoint returned 404. Subsequent inventory and secret reads confirmed complete cleanup.

The final termination log was unavailable after controller deletion.
Thus the observed proof establishes independent cleanup after an armed deadline, not an exact measured deadline-to-deletion interval.
No local termination command or Actions job performed the cleanup.
A separate failure could also cause early cleanup through the controller's `finally` block.
The result does not distinguish that path from normal deadline completion.

## Code and credential checks

The uploaded files had these SHA256 values:

- `runpod_lease.py`: `02ffa0b1eec201df6ad7b8d2998f5265f515685add1675e0674d96da3fd466f8`.
- `runpod_controller.py`: `5a50992068cbab7c38f8fd1a0e18bb7ae7853a1e420e37effdc52a49c763be36`.

The controller payload contained only the two local scripts. It did not download code from GitHub.
The GPU environment did not receive the account credential.
The controller received a secret reference, and Runpod resolved it inside that CPU container.
The secret and both Pods were absent after cleanup.
The now-unused GitHub Actions credential and SSH variable were removed. The optional workflow remains committed.

All 50 repository tests passed, including identity, retry, failure cleanup, and overlapping-rental checks.
Those checks supplement the provider observations. They do not establish CUDA correctness or deadline precision.

## Cost and limits

The quoted compute rates were $0.06/hour for the CPU controller and $0.49/hour for the GPU, plus disk charges.
Billing had not posted for either resource at the cleanup read. Zero returned records do not mean the run was free.
Reserve $0.06 for this test, covering the observed creation-to-absence window with disk allowance.
Earlier posted charges were $5.007173080695793, giving approximately $5.07 spent or reserved against the $15 total budget.
Reconcile this reserve before another rental.

A failed CPU boot can leave the inexpensive controller billing without creating a GPU.
A controller crash or provider API outage can still prevent cleanup. This is not a provider-enforced spending cap.
The launcher records exact IDs for recovery. Keep normal artifact export and manual GPU termination after a successful experiment.
The controller then handles its own cleanup.
