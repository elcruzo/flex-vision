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
