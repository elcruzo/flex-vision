# Independent GPU lease

Status: implemented, live termination proof pending.

The manually dispatched `gpu-lease.yml` workflow creates one disposable L4 Pod on a GitHub-hosted runner.
That runner starts the deadline before provisioning and terminates the Pod when the deadline arrives.
The Mac and the GPU container do not execute the watchdog.
A two-minute mode tests cleanup. A 30-minute mode supports an experiment.
No schedule, push trigger, pull-request trigger, or automatic retry starts paid compute.

The workflow accepts dispatch only on this repository's `main` branch, by its owner, on the first run attempt.
It serializes leases without canceling an existing lease. Do not queue extra rentals.
The script rejects an account with existing Pods, unavailable EU-RO-1 capacity, or an L4 price above $0.49/hour.
It creates 50 GB of disposable container disk and no persistent or network volume.
Only SSH is exposed. The API credential is not sent to the Pod.

## Credentials and authority

The repository uses the encrypted Actions secret `CPG_RUNPOD_API_KEY`.
The public SSH key is the Actions variable `CPG_SSH_PUBLIC_KEY`. The private SSH key stays on the Mac.
A repository secret is an account credential, not a Pod-scoped credential.
Restrict repository write access accordingly. Do not use this workflow for untrusted source revisions.
The workflow pins checkout to a commit and does not persist its GitHub credential.
See [GitHub's secret handling documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

Cleanup uses only the exact Pod ID returned by this run's creation response.
It also checks the `CPG_LEASE_RUN` marker. It never selects deletion targets by a name or prefix.
The ignored receipt contains the Pod ID, run ID, deadline, image, and quoted GPU rate. It contains no API credential.

## Run and verify

1. Check remaining total budget, live capacity, image, and rate before dispatch.
2. Dispatch the two-minute mode for the first live termination test.
3. Record the created Pod ID and workflow run URL.
4. Verify the deadline event, deletion, and subsequent absence through Runpod.
5. Use the 30-minute mode only after that actual cleanup proof.
6. Export and hash experiment artifacts before the deadline.
7. Terminate the Pod manually after normal completion. The watchdog then confirms absence.
8. Reconcile posted charges after cleanup.

```bash
gh workflow run gpu-lease.yml --ref main -f minutes=2 -f expires_at=UTC_UNIX_DEADLINE
gh run list --workflow gpu-lease.yml --limit 1
```

Replace `UTC_UNIX_DEADLINE` with a timestamp at most 15 minutes in the future.
A delayed runner refuses creation after that time. Queue delays do not start a later unattended rental.

The workflow job has a 45-minute timeout, leaving cleanup time after the maximum lease.
The timeout itself does not stop cloud billing. The Python `finally` block and an `always()` cleanup step perform API deletion.
Cleanup retries transient failures and requires a subsequent absence read.

## Limits

This mechanism is independent of Mac sleep. It is not a provider-enforced spending cap.
A GitHub runner failure, prolonged Runpod API outage, or forced job termination can still prevent cleanup.
An ambiguous POST response can leave a Pod without a receipt. Creation is never retried automatically.
Inspect provider state and reconcile that creation before doing anything else in that case.
Do not cancel the watchdog while the Pod exists. Use normal Pod termination instead.
GitHub can forcibly stop a canceled job. See [its cancellation behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation).

Artifact export is separate from emergency cleanup. A reached deadline discards unexported container data to bound spending.
Watch the deadline during the experiment and export incrementally.
