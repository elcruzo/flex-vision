# First remote GPU session

Status: user setup guide, checked against provider documentation on 2026-10-04.
The first NVIDIA session passed and its Pod was terminated after artifact export.
See [the session evidence](experiments/004-cuda-host.md) for the tested environment and reproduction steps.
The user authorized a $15 total development budget. The first two sessions have posted charges totaling $0.15281239314936102.
Later session charges must be added before calculating the remaining budget.
Track later sessions against the remaining total. The budget is not a per-session allowance.

## What to prepare now

Create a Runpod account and configure billing when you are ready to rent.
Choose a total session budget before starting compute.
Prepare an SSH key on this Mac and add only its public key to Runpod Credentials.
Keep the private key on the Mac. Do not paste private keys or API tokens into chat.

If you already have an appropriate SSH key, use it.
Otherwise generate a dedicated key without overwriting an existing file:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/cpg_runpod -C cpg-development
cat ~/.ssh/cpg_runpod.pub
```

Use the full `.pub` line in Runpod, not the key fingerprint.
Runpod documents public-key setup and both proxy and direct SSH methods. [SSH guide](https://docs.runpod.io/pods/configuration/use-ssh)

## Pod choices

Use a GPU Pod for this development session, not a serverless model endpoint.
A single RTX 4090 with 24 GB is a reasonable first candidate if available within your budget.
Another dedicated NVIDIA GPU can work after compatibility checks.
We do not need multi-GPU training hardware or a hosted LLM service.

Select:

- One on-demand GPU rather than an interruptible instance for the initial measurements.
- A Linux CUDA development/PyTorch template, not an Ollama or image-generation app template.
- About 32 GB host RAM and 100 GB working storage as initial allowances.
- Public-IP/full SSH access if available, for artifact transfers.
- An explicit persistent storage policy and a known teardown procedure.

Runpod's full SSH method supports SCP/SFTP. Its basic proxy SSH method does not.
Official templates can provide SSH setup, but check the selected template's connection instructions. [SSH options](https://docs.runpod.io/pods/configuration/use-ssh)

Check the live hourly price, storage charges, available GPU, and template details before deployment.
This guide does not fix a dollar amount or authorize spending.
A provider credit balance is not a substitute for a session budget and shutdown plan.

## Driver and template requirements

Prefer a CUDA 13 development environment with a compatible driver for the requested project target.
NVIDIA lists driver branch 580 or newer for CUDA 13.x minor-version compatibility.
The exact toolkit, PTX, and framework combination still needs validation. [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

A template labeled PyTorch may use an older CUDA build.
Record the image tag/digest and host driver before choosing package versions.
Do not copy the Mac requirements lock onto the GPU host as the CUDA environment specification.
Do not upgrade the host GPU driver from inside a rented container.
If the selected machine cannot support the target environment, choose another instance before extended testing.

When you reach the selection screen, the useful details to share are GPU model, hourly price, template/image, and intended session budget.
After setup, share the SSH connection command and the local key path, not the private key contents.
No Runpod API token is required for manual Pod setup and SSH development.

## First connection

Copy the exact connection command from the Pod's Connect panel.
Use its actual mapped port and local key path.
After connecting, run:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The CUDA version displayed by nvidia-smi does not establish the installed toolkit or framework build.
Check the separate outputs.
Save the template identity and these results before changing the environment.

Clone into persistent working storage:

```bash
cd /workspace
 git clone https://github.com/elcruzo/flex-vision.git
cd flex-vision
python scripts/gpu_preflight.py --output benchmark-results/preflight.json
```

The script needs the selected template's CUDA-enabled PyTorch.
It installs nothing and does not need the CPG package installed.
A missing toolkit or profiler appears in the report even if basic CUDA arithmetic succeeds.
A failed CUDA check exits nonzero. A successful smoke check does not declare the target fully ready.

## Trace preflight

After installing a compatible Nsight Systems CLI through NVIDIA's documented method, collect the small preflight:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --output=benchmark-results/preflight-trace python scripts/gpu_preflight.py --output benchmark-results/preflight-traced.json
```

Confirm flags with the installed `nsys profile --help` if its version differs.
Open the resulting trace and check the GPU work and deliberate 4096-byte device-to-host result copy.
The actual trace is the proof of tracing access. Finding an nsys executable is insufficient.
Missing hardware counters can be separate from missing CUDA tracing.
If the host restricts required tracing, resolve it or select another host before collecting benchmark evidence.
[Nsight Systems installation](https://docs.nvidia.com/nsight-systems/InstallationGuide/)

Next, establish compatible CuPy and TensorRT versions and run actual model inference.
Record that tested environment separately from the Mac lock.
The separate CPG and TensorRT experiments validate their own scopes. The arithmetic preflight does not replace them.

## End the session deliberately

Export source changes, reports, models you need to retain, and traces before deleting compute.
Check the Pod and storage state in the console after shutdown.
Stopping or terminating compute can have different data-retention consequences.
Persistent storage can continue to incur charges.
Use the provider's current lifecycle instructions for the selected storage type. [Manage Pods](https://docs.runpod.io/pods/manage-pods)

You can prepare the account and SSH access before renting.
Start paid compute when we are ready to run the CUDA experiment.


## Posted session ledger

Runpod billing was read on 2026-10-04 after the first two Pods were terminated.
These costs include the provider's recorded GPU and disk amounts.

| Experiment | Pod | Posted USD |
| --- | --- | --- |
| 004 CUDA host | 9rffjln920g60i | 0.09863918775226921 |
| 005 fused backend | hkcvv93vzorup5 | 0.054173205397091806 |

The subtotal is $0.15281239314936102. Do not interpret this as including a later unposted session.
The overall authorization remains $15 across sessions, not $15 per rental.

Experiment 006 used Pod `dsaslqanvq6ka9` at $0.49/hour from 17:18:17 to 17:35:15 UTC on 2026-10-04.
It was terminated after artifact verification. No Pods remained in the subsequent account read.
Its billing records were not yet posted. Reserve $0.15 including disk, for approximately $0.303 cumulative spend including earlier posted charges.
Reconcile this reserve with provider billing before the next rental. No persistent volume was created.

Experiment 007 used Pod `slkpedxcxg5nj0` at $0.49/hour from 17:40:08 to 17:53:16 UTC on 2026-10-04.
It was terminated after verification of all 26 artifacts. The subsequent account list contained no Pods.
Reserve $0.12 including disk for this session. Billing for 006 and 007 remained unposted at this read.
Posted charges plus the two reserves total approximately $0.423 of $15. Reconcile both reserves before the next rental.

## Shutdown control failure observed in experiment 008

The Mac session paused while a newly created Pod remained active for approximately nine hours.
A local timer used one long relative sleep. It did not fire during the pause.
Do not rely on a laptop process as an independent cloud spending limit.
Checking an absolute UTC deadline every few seconds improves behavior after resume, but cannot terminate compute while the laptop is unavailable.

The tested cluster image did not provide `runpodctl` or a Pod-scoped API key on this session.
A shell timer that invokes a missing command provides no protection. Check its executable and authorization before treating it as armed.
The public runpodctl documentation and available API/source differed on automatic termination options during this audit.
Do not assume a documented flag is supported or persisted by the provider without a verified readback.

Before the next rental, verify a provider-side expiration or an independent termination mechanism.
Set it when creating the resource so that container startup is covered.
Test its actual termination and confirm that billing resources disappear before relying on it.
Do not create another paid experiment while this protection remains unresolved.
Keep manual termination after verified artifact export as the normal completion path.

## Reconciled ledger after experiment 008

On 2026-10-05, experiment 006 posted $0.14031437516678125 and experiment 007 posted $0.10895695874933153.
Together with experiments 004 and 005, prior posted charges total $0.4020837270654738.
Replace the earlier estimates for 006 and 007 with these amounts. Do not count both estimates and posted charges.

Experiment 008 used Pod `64rp1cyu1iune8`, created at 17:56:42 UTC on October 4 and terminated at 03:12:39 UTC on October 5.
Most elapsed time was idle during a Mac-session pause. The local shutdown guard failed to terminate it during that pause.
Reserve $4.70 including disk for this entire session, giving approximately $5.10 cumulative against the $15 authorization.
The cleanup read showed partial posted charges of $4.1067960490472615 for this session.
That amount is included in the $4.70 reserve, not an additional charge.
The account had no Pods after termination. No persistent volume was created.
See [experiment 008](experiments/008-controls.md) for results and the shutdown failure.

## Posted reconciliation on 2026-10-05

Experiment 008 now has final posted GPU and disk charges of $4.605089353630319.
Replace its $4.70 reserve with that value. All five completed rentals total $5.007173080695793.
The remaining authorization is $9.992826919304207 before any later lease.
The account read contained no Pods and no new experiment rental had started.

The [direct Runpod controller](GPU_LEASE.md) now provides the default independent lease path.
Its cleanup test passed in [experiment 011](experiments/011-direct-lease.md). GitHub Actions remains optional.
Reserve $0.06 for the controller/GPU test until its billing records post.
Posted charges plus that reserve total approximately $5.07 against the $15 authorization.
Both test Pods and the temporary encrypted secret were absent after cleanup.

## Reconciliation on 2026-10-06

Experiment 011 posted $0.01869893982075155 for its CPU and GPU Pods.
Experiment 012 posted $0.07088790504167264 for its CPU and GPU Pods.
Replace their earlier reserves with those amounts. Total posted charges are $5.096759925558217 against the $15 authorization.

Experiment 014 used GPU `h1svmmclpuiqhz` and controller `7abcbhr2jj52iw` at $0.49/hour and $0.06/hour, respectively.
Both Pods and temporary secret `7xztdjwdoy` were absent after verified artifact export and cleanup.
No persistent volume was created. Billing had not posted at the cleanup read.
Reserve $0.10 including disk, giving approximately $5.20 used or reserved and $9.80 remaining.
Replace this reserve with posted charges before another rental.

## Reconciliation before experiment 016

Experiment 014 posted $0.05681364292649960 including CPU, GPU, and disk. Replace its $0.10 reserve with that amount.
Total posted spending is $5.153573568484717 against the $15 authorization.
Experiment 016 used GPU `n2x2oog9mcnytl` and controller `nvah9ciyl9tjqq`.
Both Pods and temporary secret `thh9dns536` were absent after verified export and cleanup.
Reserve $0.10 including disk for experiment 016 until billing posts.
Posted spending plus that reserve is approximately $5.25, leaving approximately $9.75.

## Experiment 017 reserves

The subsequent billing read still reported $5.153573568484717. Experiment 016 remained unposted, with its $0.10 reserve retained.
Controller `tkgx9yijdp4iuj` disappeared during a session pause longer than the lease allowance.
Its GPU creation was not observed. Provider reads confirmed no remaining Pods or temporary secret `zik53spr42`.
Reserve $0.35 for that attempt until its charges can be reconciled.

The successful retry used CPU controller `s704oi3w6pj60p` and GPU `jm0qabsvc1bfg2`.
Both measured revisions used that lease. Verified export preceded GPU termination.
Both Pods and temporary secret `3rqroevu1e` were absent after cleanup. Reserve $0.10 for this retry.
Posted spending plus these three reserves is approximately $5.70, leaving approximately $9.30 of the $15 authorization.
Replace reserves with corresponding posted charges before the next rental. Do not count both.

## Reconciliation before experiment 018

The October 7 billing read reports $5.533818361645899 across 17 Pods.
The account inventory contained no active Pods before the new lease.
Experiment 016 posted $0.045833331532776356 for its GPU and controller.
The successful experiment 017 retry posted $0.055046326908268384.
The earlier controller `tkgx9yijdp4iuj` posted $0.03065008862540708.
Billing also lists GPU `hv9r4melf995m4` at $0.2487150460947305 in the same period.
Its creation receipt was not captured during the session pause, so attribution to that attempt remains unverified.
Include it in the budget nevertheless. These amounts replace the prior three reserves, without double counting.
The remaining authorization before experiment 018 is $9.466181638354101.

## Reconciliation after experiment 018

The billing read through October 8 reports $5.927725856261532 across 23 Pods.
The account inventory contains no Pods. This total includes all three experiment 018 attempts:

| Attempt | GPU | Controller | Combined posted cost |
| --- | --- | --- | ---: |
| Initial sustained run | `c4d7dsr6plg1i1` | `tjjpwmqgkv9bx7` | $0.27893270948334248 |
| Failed container startup | `sh15th8cxptfxv` | `lcr83j97kot9zr` | $0.013208999997004866 |
| Revised sustained run | `dv5dz5vttj0dae` | `f5gj60n0xexvw7` | $0.10176578513528511 |

These posted amounts replace reserves for those attempts. No reserve remains for a running lease.
The remaining authorization is $9.072274143738468 of the original $15.
The earlier $5.81 progress figure excluded the failed startup and revised run. Use this reconciled total instead.
