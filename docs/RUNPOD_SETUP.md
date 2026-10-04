# First remote GPU session

Status: user setup guide, checked against provider documentation on 2026-10-04.
The first NVIDIA session passed and its Pod was terminated after artifact export.
See [the session evidence](experiments/004-cuda-host.md) for the tested environment and reproduction steps.
The user authorized a $15 total development budget. This session used approximately $0.10, pending the billing ledger.
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
The CPG CUDA backend is still the next implementation step, not something the preflight claims to validate.

## End the session deliberately

Export source changes, reports, models you need to retain, and traces before deleting compute.
Check the Pod and storage state in the console after shutdown.
Stopping or terminating compute can have different data-retention consequences.
Persistent storage can continue to incur charges.
Use the provider's current lifecycle instructions for the selected storage type. [Manage Pods](https://docs.runpod.io/pods/manage-pods)

You can prepare the account and SSH access before renting.
Start paid compute when we are ready to run the CUDA experiment.
