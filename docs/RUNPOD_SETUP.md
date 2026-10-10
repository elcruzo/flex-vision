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

## Experiment 019 active reserve

The user requested the continuous 30-minute soak. Reserve $0.65 including disk for its 60-minute direct-controller lease.
GPU `mlrt00axvzkm05` costs $0.49/hour. Controller `41k6ky75r5ao19` costs $0.06/hour.
The controller armed the GPU deadline at 2026-10-08 02:53:50 UTC, independently of the Mac.
The temporary secret is `txilc0n911`. Verify both Pods and that secret are absent after cleanup.
Posted charges plus this reserve total $6.577725856261532, leaving $8.422274143738468 until reconciliation.

The initial thermal check failed. Both Pods and secret `txilc0n911` were absent after verified export and cleanup.
Retain its reserve until billing posts. The retry uses controller `gu2mqlrfefs9y0` and secret `jrdmqti3ir`.
Reserve a further $0.65 for the retry, giving $7.227725856261532 used or reserved and $7.772274143738468 remaining.

## Experiment 019 reconciliation and automatic retry

The first attempt posted $0.2547739035253471. The idle retry posted $0.5576251531092567.
Both reservations are replaced by these charges. Total posted spending is $6.740124912896135, leaving $8.259875087103865.
The retry expired during a session pause before SSH submission. No measured soak ran on that lease.
Both Pods and its temporary secret were absent at the subsequent read.
The next attempt uses automatic SSH submission and export under the same independent controller.
Reserve $0.65 for it, giving $7.390124912896135 used or reserved and $7.609875087103865 remaining.

## Experiment 019 completed reconciliation

The successful automatic lease used GPU `czpobr6j1rnf3t` and controller `y8yfpgzwyckpn8`.
Its GPU and disk posted $0.3577198227867484. Its controller and disk posted $0.04372778411197942.
The combined $0.4014476068987278 replaces the $0.65 reserve.
All 15 artifacts were hash-verified before termination. Both Pods and secret `tgqlhnjg8x` were absent after cleanup and at a later read.
Total posted spending is $7.141572519794863 across 29 Pods. The remaining authorization is $7.858427480205137.
There are no active Pods or outstanding rental reserves at this reconciliation.

## YOLO experiment reservation

The next fixed YOLO comparison reserves at most $1.10 within the existing $15 total authorization.
The live catalog reported no CUDA 13 L4 capacity and a secure L4 rate of $0.59/hour.
An RTX 4090 was available at $0.89/hour in EU-CZ-1 and EU-RO-1.
The controller now accepts that explicitly selected GPU with a $0.89/hour bound.
Its CPU bound remains $0.06/hour. The one-hour independent expiration and early cleanup remain mandatory.
The existing default L4 bound remains $0.49/hour. This change does not authorize overlapping rentals.
Treat the new GPU as a separate measurement platform. Do not compare absolute latency across these rentals.
Reconcile the reservation against posted charges after termination.

### October 9 reconciliation and active retry

The first two YOLO controllers ended before GPU readiness. Both controllers and temporary credentials were absent afterward.
Their posted charges were $0.00015130000247154385 and $0.00015809999604243785, with no GPU charges.
These charges replace their reservations. Total posted spending is $7.141881919793377 across 31 Pods.
The remaining authorization before the next lease is $7.858118080206623.

Reserve $1.10 for the current Romania retry, including the bounded controller and disk costs.
GPU `w45z3wghznchc0` costs $0.89/hour. Controller `mebwvjzisbkdci` costs $0.06/hour.
Its independent deadline is October 9 at 05:02:12 UTC (01:02:12 Eastern).
The controller removes secret `otg2y0ntsu` after termination. Verify all three resources are absent before closing the lease.
Posted spending plus this reservation is $8.241881919793377, leaving $6.758118080206623 unreserved.

The first GPU execution failed the dense-output check before timing. Ten artifacts were exported and hash-verified.
GPU `w45z3wghznchc0`, controller `mebwvjzisbkdci`, and secret `otg2y0ntsu` were verified absent at 04:05:51 UTC.
Its billing has not posted yet. Retain its $1.10 reserve until reconciliation.
Reserve another $1.10 for the corrected-model retry. Posted spend plus both reserves is $9.341881919793377.
This leaves $5.658118080206623 unreserved within the $15 total authorization.

The corrected-model rental also stopped before timing because of cross-engine background box differences.
All 19 files were exported and verified. GPU `qjps1uowdsbdrx`, controller `qqeqvzrluw0ujb`, and secret `boxxq1i7jh` were removed.
The complete cleanup check finished at 04:10:58 UTC on October 9.
Its charge is not posted yet. Retain both pending $1.10 reserves and add $1.10 for the acceptance-revision-2 execution.
Posted spend plus these conservative reserves is $10.441881919793377, leaving $4.558118080206623 unreserved.
The rate and one-hour expiration remain unchanged. Replace reserves with actual charges when the provider posts them.

The acceptance-revision-2 rental ended after another numerical failure. All 16 artifacts were exported and verified before cleanup at 04:17:34 UTC.
The revision-3 controller `9eut20ybvqoyxj` then stalled during startup without allocating a GPU.
It and secret `f6rml6ieww` were explicitly removed at 05:01:28 UTC. The account inventory was empty afterward.
The next controller uses EU-CZ-1 with its GPU in EU-RO-1. No GPU allocation from the stalled attempt is reused.
Retain a conservative $1.10 reserve for each unposted attempt: four ended attempts and the current retry, $5.50 total.
Posted spending plus these reserves is $12.641881919793377, leaving $2.358118080206623 unreserved.
Each ended attempt will replace its reserve with its actual charge after billing posts.

### Revision 4 reservation

The provider now reports $7.248022670855789 posted across 35 Pods.
The first two GPU attempts posted $0.10614075106241246, including their controllers. These charges replace their two reservations.
Three ended attempts remain unposted. Retain their conservative $3.30 combined reservation.
The revision 4 attempt reserves another $1.10. Posted spending plus reservations is $11.648022670855789.
This leaves $3.351977329144211 unreserved within the $15 authorization.
The account had no active Pods before this attempt. The RTX 4090 rate remains $0.89/hour.
Controller `94k9vajfheu4g5` uses EU-CZ-1 and requests its GPU in EU-RO-1.
Its independent lease is 60 minutes. Verify GPU, controller, and temporary credential removal after export.

### Completed comparison and cached-metadata follow-up

The revision 4 run exported 61 verified files, then removed its GPU at 05:20:52 UTC on October 9.
Controller `94k9vajfheu4g5` and secret `o3i24gpxr8` were absent at 05:21:03 UTC.
The provider now reports $7.314297882954634 posted across 37 Pods.
The revision 2 attempt posted $0.06627521209884435, replacing its reservation.
The stalled controller, revision 3 attempt, and revision 4 comparison remain unposted. Retain $3.30 for those ended attempts.
Reserve $1.10 for the cached-metadata follow-up under controller `8qkr7rggb9hvu5`.
Posted spending plus all conservative reserves is $11.714297882954634. This leaves $3.285702117045366 unreserved.
The account inventory was empty before creation. The GPU rate remains $0.89/hour with independent 60-minute expiration.

The cached-metadata run exported and hash-verified 67 files with exit code 0.
GPU `ieodtkhocuwbqm` was absent after termination at 05:28:32 UTC.
Controller `8qkr7rggb9hvu5` and secret `tf2pkengxk` were absent at 05:28:54 UTC and at a later account read.
There are no active Pods. The latest posted total remains $7.314297882954634.
Four ended attempts remain unposted. Retain $4.40 until actual charges replace those reservations.
Posted spending plus conservative reservations remains $11.714297882954634, with $3.285702117045366 unreserved.

### Detector repeatability and bounded-load reservation

The next lease uses controller `0rogkq54tgxx9y` with its GPU in EU-RO-1.
The account inventory was empty before creation. GPU and CPU rates remain $0.89/hour and $0.06/hour.
Reserve $1.10 within the existing $15 authorization. The independent expiration remains 60 minutes.
The latest posted total is $7.314297882954634. Retain $4.40 for four ended attempts whose billing remains unposted.
With this new reserve, posted spending plus conservative reservations is $12.814297882954634.
This leaves $2.185702117045366 unreserved. Reconcile actual charges after termination rather than treating reservations as spending.

The first detector load attempt exported 72 verified artifacts and failed its allocation guard.
GPU `ywq5838hv6uldi` was removed at 05:50:04 UTC on October 9. Controller and secret removal passed at 05:50:15 UTC.
Its charge remains unposted. Retain its $1.10 reserve and add $1.10 for the checked-warmup retry.
The latest posted total remains $7.314297882954634. Six conservative $1.10 reserves now cover five ended unposted attempts and this retry.
Posted spending plus reservations is $13.914297882954634, leaving $1.085702117045366 unreserved within the $15 authorization.

### Complete-matrix diagnostic lease

The checked-warmup attempt exported 77 verified files, then removed its GPU at 06:01:23 UTC on October 9.
Controller `zu350lrr6rgzs0` and secret `mab7iw4sl4` were absent at 06:01:34 UTC. A later read found no active Pods.
The latest provider posted total is $7.315671662643581. Keep all six earlier conservative reservations until reconciliation.
The final matrix and auxiliary-resource diagnostic use a shorter 45-minute independent GPU lease.
The GPU bound is $0.89/hour and the controller bound is $0.06/hour.
Reserve $0.85 for this lease, including startup, cleanup, and ephemeral storage headroom.
Posted spending plus earlier reserves and this new reserve is $14.765671662643581, below the authorized $15.
The local export allowance is also shortened to preserve five minutes before the GPU deadline.
No overlapping GPU is authorized. Export and verify evidence before termination.

### Completed detector matrix: final cleanup and billing

The final lease exported 81 hash-verified artifacts. GPU `is89917laka93x` was removed at 06:22:38 UTC on October 9.
A fresh read at 06:32:54 UTC found no Pods. Controller `q6c8jo3c8obv8c` and secret `funi1dcb70` were absent.
The posted pod-billing total is now $7.545256926736329. This is posted spending, not the final invoice.
Keep the six earlier $1.10 reservation ceilings and the final $0.85 ceiling until charges are complete.
Offset $0.23095904378169507 already posted for reserved Pods `69as2koczlkkmr`, `f80daayqa3d5sh`, `ldesb98veqbhlk`, `94k9vajfheu4g5`, `ieodtkhocuwbqm`, and `8qkr7rggb9hvu5`.
This avoids counting their posted charges twice without releasing their remaining reservation headroom.
Posted spending plus remaining conservative reservations is $14.764297882954633, within the $15 authorization.
No new rental is scheduled. Reconcile pending charges before another paid experiment.

A later October 9 billing read reports $7.685433012959038 in posted pod charges across 48 records.
New records include the two ended detector attempts. Their charges still may be incomplete.
Offset the posted amounts for reserved Pods without releasing the remaining reservation ceilings.
The conservative total remains about $14.77 against the $15 authorization.
The 30-minute detector soak harness is prepared locally. No new rental was created.

The latest October 9 read reports $8.033368132837495 in posted pod charges across 50 records.
Both final-matrix resource IDs now have billing records. Posted records can still be incomplete.
Keep the remaining reservation headroom until reconciliation. No new rental was created.

### Lifetime-bounded reconciliation for the detector soak

Billing still reports $8.033368132837495 posted. A final invoice is not required to bound already terminated compute.
Replace the seven remaining full-lease reservations with actual-lifetime upper bounds.
Each run marker records a timestamp before controller creation. Each cleanup record bounds both controller and GPU lifetimes.
Use an additional 60 seconds for timing headroom, plus $0.10 per attempt for storage and delayed adjustment.
Charge both GPU and controller for that entire interval, even though the GPU starts later.
Use $0.89/hour GPU and $0.06/hour CPU caps. The stalled 020f attempt allocated only its CPU controller.
For 020g, use the subsequent 020h creation time as the cleanup upper bound because the intervening inventory was empty.
For 021c, use the later 06:32:54 UTC empty-inventory read rather than its earlier cleanup.
Keep any posted amount exceeding its lifetime bound instead of reducing it.

| Attempt | Lifetime bound, including headroom | Total reservation ceiling |
| --- | ---: | ---: |
| 020f | 316 seconds, CPU only | $0.105267 |
| 020g | 814 seconds | $0.314806 |
| 020h | 434 seconds | $0.214528 |
| 020i | 413 seconds | $0.208986 |
| 021 | 525 seconds | $0.238542 |
| 021b | 697 seconds | $0.283931 |
| 021c | 1454 seconds | $0.483694 |

Posted amounts already covered by these ceilings total $0.7190702498828614.
Retain another $1.1306825278949164 against them. This avoids double counting their posted charges.
Reserve $1.10 for one new independently terminated 60-minute RTX 4090 lease.
The conservative cumulative bound becomes $10.26405066073241, below the authorized $15.
The live catalog reports secure RTX 4090 at $0.89/hour with HIGH availability in EU-RO-1.
The pre-creation account inventory is empty. No overlapping rental is authorized.
The next experiment checks caller-owned output through TensorRT before the unchanged default-output soak.

The first soak dispatch failed before CUDA setup because the SSH shell lacked the Pod's deadline environment variable.
GPU `jdq1fhfhe339we` was removed at 08:01:34 UTC on October 9.
A subsequent API read found no Pods and confirmed secret `l3wvbbvzcd` absent.
Retain that attempt's full $1.10 reservation until reconciliation.
The coordinator now explicitly exports the validated Pod deadline into the remote shell.
It creates the result directory for early failures and checks controller cleanup on exception paths.
Reserve another $1.10 for the corrected retry. The conservative cumulative bound is $11.36405066073241.
No numerical, reuse, or soak acceptance came from the failed dispatch.

### Completed detector soak and reuse lease

The corrected run exported 78 hash-verified artifacts. GPU `2w8tzsvt2wr2wm` was removed at 08:44:52 UTC on October 9.
Controller `rzexvi6tq4jb0p` and secret `7tx1ljwerw` were absent at 08:45:13 UTC.
A fresh read at 14:05:48 UTC found no Pods and confirmed both absent.
Provider billing now reports $8.70408385734845 posted across 54 records for 53 unique Pods.
Retain $2.659966803383961 of reservation headroom after offsetting posted amounts for the reserved resources.
The conservative cumulative total remains $11.364050660732412 within the authorized $15.
This includes both the failed startup and successful soak lease. Reservations are not actual spending.
No new lease is active or scheduled. See experiment 022 for acceptance and retained evidence.

### Output-reuse matched comparison reservation

The live preflight found no Pods and reports secure RTX 4090 at $0.89/hour with LOW availability in EU-RO-1.
Reserve $0.65 for one independently terminated 30-minute lease, including the CPU controller capped at $0.06/hour and storage headroom.
The posted total remains $8.70408385734845. Keep all existing delayed-charge reservation headroom.
The conservative cumulative bound with this new reservation is $12.014050660732413, within the authorized $15.
The experiment regenerates controls, verifies output reuse, and measures fresh versus caller-owned output through the same TensorRT consumer.
No overlapping rental is authorized. Verify exported evidence and cleanup before accepting the result.

The matched experiment completed and exported 71 hash-verified files. All 40,000 reuse-comparison frames passed.
GPU termination was verified at 16:22:13 UTC on October 9. Controller and secret cleanup completed at 16:22:24 UTC.
The fresh 16:22:52 UTC inventory was empty. Posted billing remained $8.70408385734845.
Retain the $0.65 reservation and the $12.014050660732413 conservative cumulative bound until charges reconcile.
No rental remains active. See experiment 023 for measured latency and allocation limits.

### Completed synchronous detector stream lease

Experiment 024 used one independently terminated 30-minute RTX 4090 lease at $0.89/hour and CPU controller at $0.06/hour.
All 48 real inference executions passed. The coordinator exported 69 hash-verified files.
GPU cleanup completed at 23:03:53 UTC on October 9. Controller and secret cleanup completed at 23:04:14 UTC.
A fresh 23:04:18 UTC inventory was empty. No rental remains active.
Posted billing was $8.783250520227739. Retain old headroom and the new $0.65 reservation.
Without offsetting the posted increase, the conservative cumulative bound is $12.743217323611702, within $15.
See experiment 024 for scope and evidence. Reconcile before another paid rental.

### Async submission correctness attempts

The first attempt failed before async inference because Torch 2.9.1 streams lacked __cuda_stream__.
It exported 67 hash-verified files and removed GPU 4q0ttle5butxnf at 23:49:37 UTC on October 9.
Controller and secret cleanup completed at 23:49:48 UTC. The subsequent inventory was empty.
The corrected attempt passed all 48 async-submit TensorRT executions and exported 69 hash-verified files.
GPU 0e3cultzi45uw0 was removed at 23:58:51 UTC. Controller and secret cleanup completed at 23:59:03 UTC.
The fresh 23:59:19 UTC inventory was empty. No rental remains active.
Posted billing remained $8.783250520227739. Retain both $0.65 reservations and all previous delayed-charge headroom.
The conservative cumulative bound is $14.043217323611703 within $15. Reconcile before another rental.
See experiment 025 for the failure, correction, scoped correctness, and remaining acceptance.

### Delayed-consumer lease with missed export

Experiment 026 submitted the scenario but exported no result bundle after a local session pause.
At 02:19:25 UTC on October 10, the GPU, controller, and temporary secret were independently absent.
No rental remains active. The delayed-consumer scenario remains unvalidated.
Posted billing is $9.770127805024458. Newly posted reserved resources total $0.8992737082371605.
Offset those charges against reservations to avoid double counting, while retaining existing ceiling headroom.
The conservative cumulative bound is $14.780820900171262 within $15. Reconcile before another paid retry.
The coordinator now checks the independent wall deadline before polling and export.
See experiment 026 for the missing evidence and session-continuity limits.

### Lifetime-bounded reconciliation after experiment 026

The fresh Pod inventory is empty. The October 2–11 UTC billing-window read still reports $9.770127805024458 posted.
Replace only four prior $0.65 reservations using their recorded run-start and confirmed-absence times.
Bound both compute rates over that entire interval, add 60 seconds, and retain $0.10 storage headroom per attempt.
Use the larger of posted charges and that lifetime bound. Keep all other reservation ceilings unchanged.

| Attempt | Bounded seconds | Replacement ceiling |
| --- | ---: | ---: |
| 023 | 516 | $0.23616666666666666 |
| 024 | 427 | $0.21268055555555554 |
| 025 failed | 1154 | $0.4045277777777778 |
| 025 corrected | 610 | $0.2609722222222222 |

These bounds use the historical $0.89/hour GPU and $0.06/hour controller rates, not a new provisioning quote.
The conservative cumulative bound falls from $14.780820900171262 to $13.295168122393484.
A prospective $0.65 reservation would bring it to $13.945168122393484, within the existing $15 authorization.
No new rental was created or reserved by this reconciliation. Recheck capacity and rates before dispatch.
See [the calculation record](experiments/data/budget-after-026.json) for exact resource identities and posted offsets.
The longer missed-export attempt 026 retains its full ceiling. No earlier delayed-charge headroom was removed.

### Focused delayed-consumer retry reservation

The October 10 preflight found no Pods. Posted Pod billing remains $9.770127805024458.
The live catalog reports secure RTX 4090 HIGH capacity in EU-RO-1 at $0.89/hour.
Two cpu3c vCPUs have HIGH capacity in EU-CZ-1 at $0.06/hour.
Reserve $0.65 for one independently terminated 30-minute session, including storage headroom.
The conservative cumulative bound becomes $13.945168122393484 within the original $15 authorization.
Regenerate all six numerical controls, then run the delayed-consumer gate.
Only run the matched serial comparison if that gate passes and enough export allowance remains.
Use ordinary local SSH export. It retains its Mac-session dependency. No bucket is required or created.

Experiment 027 completed with 40 hash-verified exported files and exact GPU/controller/secret cleanup.
Retain its $0.65 reservation. The conservative cumulative bound remains $13.945168122393484.
No rental remains active. See experiments/027-delayed-and-async.md before choosing the next GPU gate.

### Async diagnostic capture reservation

The October 10 preflight found no Pods. Posted Pod billing remains $9.770127805024458.
The live catalog reports secure RTX 4090 HIGH capacity in EU-RO-1 at $0.89/hour.
Two cpu3c vCPUs have HIGH capacity in EU-CZ-1 at $0.06/hour.
Reserve $0.65 for one independently terminated 30-minute diagnostics session.
The conservative cumulative bound becomes $14.595168122393484 within the original $15 authorization.
Run fresh numerical controls, pending-consumer checks, and separate async trace/allocator diagnostics.
Retain results even when the capture or allocation gate fails. Keep ordinary local SSH export.
No cloud storage or persistent volume is required.

Experiment 028 exported 43 verified files and confirmed exact GPU/controller/secret cleanup.
The fresh inventory is empty. Keep its $0.65 reservation until reconciliation.
The conservative cumulative bound remains $14.595168122393484. Reconcile before another rental.

### Reconciliation and CAI inference reservation

The October 10 live Pod inventory is empty. Posted Pod billing remains $9.770127805024458.
No posted row exists yet for the four experiment 027/028 Pods.
Replace only their two $0.65 reservations with recorded full-lifetime ceilings.
Apply both $0.89/$0.06 compute rates for the entire run-to-confirmed-absence interval, plus 60 seconds and $0.10 storage each.
The reconciled conservative cumulative bound is $13.706279233504596. All other ceilings remain unchanged.
See experiments/data/budget-after-028.json for exact identities and calculation.

The live catalog reports secure RTX 4090 HIGH capacity in EU-RO-1 at $0.89/hour.
Two cpu3c vCPUs have HIGH capacity in EU-CZ-1 at $0.06/hour.
Reserve $0.65 for one independently terminated 30-minute CAI inference session.
The cumulative conservative bound becomes $14.356279233504596 within the existing $15 authorization.
Regenerate six numerical controls and run the 144-case CAI protocol.
Retain failed evidence. No trace or performance acceptance follows from this correctness run.

Experiment 029 exported its failed CAI harness report and verified exact GPU/controller/secret cleanup.
No CAI execution passed. Retain the $0.65 reservation until reconciliation.
The conservative bound remains $14.356279233504596. The fresh inventory is empty.

### Corrected CAI retry reservation

The live Pod inventory is empty. Posted Pod billing remains $9.770127805024458.
Replace only the ended 029 reservation using its 261-second full lifetime, both compute rates, 60 seconds, and $0.10 disk headroom.
Its ceiling is $0.18470833333333334. The reconciled cumulative bound is $13.890987566837929.
Keep all other ceilings. See experiments/data/budget-after-029.json.
Secure RTX 4090 HIGH capacity is available in EU-RO-1 at $0.89/hour.
Two cpu3c vCPUs have HIGH capacity in EU-CZ-1 at $0.06/hour.
Reserve $0.65 for one independently bounded 30-minute corrected CAI session.
The conservative cumulative bound becomes $14.540987566837929, within the original $15 authorization.
Require all 144 scenarios. Preserve failures and verify export and exact cleanup.

Experiment 030 resources are absent. Its lifetime bound replaces only its $0.65 reservation.
The cumulative conservative bound including one corrected 031 retry is $14.781640344615706.
See experiments/data/budget-after-030.json. The original $15 authorization remains unchanged.

Experiment 031 resources are absent. Its full-lifetime bound replaces only its $0.65 reservation.
Including one CAI transfer-capture reservation, the conservative cumulative bound is $14.98270978906015.
See experiments/data/budget-after-031.json. No additional budget is authorized.

Experiment 032 is absent and now has a full-lifetime replacement bound.
One 15-minute stencil gate is reserved at $0.40, including compute, controller boot, and storage headroom.
The cumulative conservative bound is $14.967029233504595 within the original $15 authorization.
Newly posted 027 charges are already covered by its retained ceiling. See experiments/data/budget-after-032.json.
