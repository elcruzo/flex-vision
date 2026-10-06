# GPU leases without a CI dependency

GitHub Actions is optional. It is not required to develop, test, or run CPG.
The default rental path uses a disposable Runpod CPU controller.
Its independent cleanup test passed. See [experiment 011](experiments/011-direct-lease.md).
The controller creates the GPU Pod, enforces its deadline, and terminates both Pods.
It runs independently of the Mac and the GPU container.

## Direct Runpod path

Set `RUNPOD_API_KEY` and `CPG_SSH_PUBLIC_KEY` in the local process environment through your credential manager.
Do not paste credentials into command arguments, reports, or source files.
Then launch the two-minute cleanup test:

```bash
python3 scripts/runpod_controller.py launch --minutes 2 --receipt .cache/runpod/controller-test.json
```

After a verified cleanup test, use `--minutes 30` and a new receipt path for an experiment.
The launcher exits after recording the CPU Pod ID. The Mac does not execute the timer.
The GPU deadline starts before GPU provisioning, so image startup uses the same lease allowance.
The CPU controller refuses to create a GPU if its own startup is more than ten minutes late.
A failed CPU boot still requires manual cleanup. No GPU is created until the controller executes.

The CPU request uses two `cpu3c` vCPUs, two GB of disposable disk, and no exposed ports or persistent volume.
The current price guard is $0.06/hour for the CPU controller and $0.49/hour for one L4, excluding disk charges.
Read current capacity, rates, and remaining total budget before every rental. These limits do not authorize new spending.
The controller uses the official `python:3.12-slim-bookworm` image because it needs Python and CA certificates, not CUDA.
Only the local lease/controller scripts are included in its startup payload. It does not clone a repository or install experiment dependencies.

### Credential handling

The launcher stores a temporary encrypted Runpod secret and passes its reference to the CPU controller.
The controller receives the account credential as `CPG_CONTROL_API_KEY`. The GPU Pod never receives that credential.
The secret is removed after GPU cleanup, before controller self-deletion.
The local receipt records its exact ID for cleanup if controller boot fails. It contains no credential value.
This remains an account credential inside the trusted CPU process, not a Pod-scoped token.
Do not execute model code, downloaded scripts, or user workloads on the controller.
See [Runpod secret creation](https://docs.runpod.io/api-reference-v2/account/create-a-secret) and [Pod environment variables](https://docs.runpod.io/pods/templates/environment-variables).

### Verify and end a lease

Read the controller logs to record the `armed` event, exact GPU ID, and UTC deadline.
Export experiment artifacts incrementally and verify hashes before terminating the GPU normally.
The controller detects that deletion, removes the temporary secret, and deletes itself.
For the initial test, let the deadline expire and verify that both Pods and the secret disappear.
Self-deletion can interrupt the final API response, so an external absence read is required.
Do not delete the controller first while its GPU is still running.

A controller process failure, failed boot, or Runpod API outage can still require manual cleanup.
The controller is independent of Mac sleep, not independent of Runpod infrastructure.
Neither path is a provider-enforced budget cap. Keep the existing total-spend ledger.
An ambiguous creation response needs reconciliation before retrying. Never select deletion targets only by a name.

## Optional GitHub Actions path

The earlier Actions attempt did not receive a runner. See [experiment 010](experiments/010-lease-preparation.md).
The account's Actions availability does not block the direct Runpod path.

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

### Actions credentials and authority

To opt in, configure the encrypted Actions secret `CPG_RUNPOD_API_KEY` and the Actions variable `CPG_SSH_PUBLIC_KEY`.
The earlier copies were removed when Actions became optional. The workflow does not need to be enabled for direct rentals. The private SSH key stays on the Mac.
A repository secret is an account credential, not a Pod-scoped credential.
Restrict repository write access accordingly. Do not use this workflow for untrusted source revisions.
The workflow pins checkout to a commit and does not persist its GitHub credential.
See [GitHub's secret handling documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

Cleanup uses only the exact Pod ID returned by this run's creation response.
It also checks the `CPG_LEASE_RUN` marker. It never selects deletion targets by a name or prefix.
The ignored receipt contains the Pod ID, run ID, deadline, image, and quoted GPU rate. It contains no API credential.

### Run and verify the optional workflow

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

### Actions limits

This mechanism is independent of Mac sleep. It is not a provider-enforced spending cap.
A GitHub runner failure, prolonged Runpod API outage, or forced job termination can still prevent cleanup.
An ambiguous POST response can leave a Pod without a receipt. Creation is never retried automatically.
Inspect provider state and reconcile that creation before doing anything else in that case.
Do not cancel the watchdog while the Pod exists. Use normal Pod termination instead.
GitHub can forcibly stop a canceled job. See [its cancellation behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation).

Artifact export is separate from emergency cleanup. A reached deadline discards unexported container data to bound spending.
Watch the deadline during the experiment and export incrementally.
