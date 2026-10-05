# Inference attribution controls

Result: the downstream timing difference depends on preceding work, rather than the inference input pointer alone.
Common memory conditioning largely removed that difference. This supports a memory-state effect, consistent with cache behavior.
It does not prove a specific cache mechanism without counters or more targeted experiments.

## Controls and validation

The first five controls ran at clean revision `7362917`. The memory-conditioned control ran at `9b6209b`.
All controls used the same engine on the same L4 during this session.
The newer revision adds the conditioning option. The option is disabled by default.

Each control collected 1,200 samples per candidate across four alternating-order repeats, with 50 warmups before each candidate block.
The six controls therefore recorded 14,400 timed inferences.
These short attribution runs do not replace experiment 007's 10,000-sample comparison or establish tail-latency performance.
The four-photo detector validation passed before the controls.
Every control also validated both real preprocessing paths against independent pixels and actual detections before timing.
All 43 local checks passed.

| Control | Timed preprocessing | TensorRT input | Dense output allocation |
| --- | --- | --- | --- |
| Live | Respective candidate | Respective output | Fresh each call |
| Fixed input | Respective candidate | Same retained pointer | Fresh each call |
| No preprocessing | Neither candidate | Same retained pointer | Fresh each call |
| Fixed input, reused outputs | Respective candidate | Same retained pointer | Reused |
| No preprocessing, reused outputs | Neither candidate | Same retained pointer | Reused |
| Conditioned | Respective candidate, then common 256 MiB write | Same retained pointer | Reused |

The scratch kernel writes varied values and completes before inference.
Its cost belongs to the reported preprocessing interval. That interval cannot represent normal preprocessing performance in this control.
Output reuse is experimental and serial. The next inference overwrites the dense output tensors after the preceding decoder completes.
The CPG library's fresh-output ownership contract is unchanged.

## Results

Values are median per-repeat reductions in p50, in percent. Positive values favor the CPG label.
The no-preprocessing labels run equivalent code, so their difference is a control observation, not a product gain.

| Control | Network interval | Decode/NMS interval | Complete host latency |
| --- | ---: | ---: | ---: |
| Live | 8.072 | 5.689 | 6.468 |
| Fixed input | 7.905 | 5.562 | 6.328 |
| No preprocessing | 0.025 | -0.028 | -0.025 |
| Fixed input, reused outputs | 7.858 | 4.243 | 5.056 |
| No preprocessing, reused outputs | -0.068 | -0.061 | -0.049 |
| Conditioned | -0.488 | 0.077 | 0.766 |

With common memory conditioning, network reductions ranged from -0.554% to -0.311% across repeats.
Decoder reductions ranged from -0.173% to 0.737%.
These ranges describe the observed repeats, not confidence intervals.

Holding the input pointer fixed did not remove the effect. Removing preprocessing removed nearly all of it.
Reusing dense outputs did not remove the network difference, although the decoder difference became smaller.
The common memory write reduced both downstream differences to less than 1% at the median.
Preceding memory traffic is therefore a stronger explanation than different input values, input addresses, or dense-output allocation alone.
Clock, cache, allocator, and dispatch interactions are not fully separated by this experiment.

## Consequence for the project

Keep complete-graph measurements. An operator can affect a later stage even when that stage's implementation and input values are unchanged.
Do not describe the earlier complete latency improvement as a direct isolated-kernel speedup.
The measured preprocessing improvement remains scoped to the tested graph and baseline.
The complete two-workload success gate, vendor comparisons, Jetson, and ROS remain pending.

Do not add the scratch write to production. It deliberately changes memory state and adds work.
Next, preserve these controls in the benchmark harness and establish a second inference workload.
Before another paid rental, resolve and verify independent cloud expiration as described below.

## Shutdown failure and budget

The Pod was created before a long Mac-session pause. It remained running for approximately nine hours before work resumed.
The local timer used one long sleep and did not fire during that pause.
No Pod-side timer had been installed before the pause because SSH was not yet available.
This was a shutdown-control failure, not useful benchmark compute time.

After resume, the attempted Pod-side timer could not run: this image lacked `runpodctl` and a Pod-scoped API key.
The local replacement checked an absolute UTC deadline every five seconds, but still depended on the Mac being available.
The run was completed and manually terminated after verified artifact export.
A laptop timer is not a cloud spending cap. Future rentals require a verified independent termination mechanism before benchmark work starts.
The user authorization remains $15 total across all sessions.

## Exported evidence and final cleanup

All 35 exported artifacts matched their remote SHA256 values.
All six summaries were independently recomputed on the Mac and matched the GPU-host summaries.
All preprocessing tensors matched the independent reference exactly in each validation.
Local bundle: `benchmark-results/iteration-008-controls/`. Large samples and models remain outside Git.

| Raw sample file | SHA256 |
| --- | --- |
| live/samples.csv | 02895e270c405fafa0f4fbddac221bfa043deeb1e00f431393b2ee1dd008b3df |
| fixed/samples.csv | 2b20ca5d4d7b4e236cd9319ff2c8bae75ffb51b70a879d6009b2f7816672eff9 |
| empty/samples.csv | 322ea9ae052de7a9dad6df9ed252b665cfad6e732c958d3ae078709d23634bc7 |
| fixed-reuse/samples.csv | fb46ec48fec05e844a15f8edb2612710fcd7eef462dc37ea6989561415ede18b |
| empty-reuse/samples.csv | b7a28f02e71d6ed0a5851c6705a7c5024416861befba3a4831c9c9be2b44c6b3 |
| conditioned/samples.csv | c474db63540eb9933b69613c9f28f47e46f2c93d6526d16885c9796947478531 |

Pod `64rp1cyu1iune8` ran at $0.49/hour in EU-RO-1 with 50 GB temporary disk and no persistent volume.
Creation: 2026-10-04 17:56:42 UTC. Termination: approximately 2026-10-05 03:12:39 UTC.
Elapsed compute cost is approximately $4.54, excluding disk. Reserve $4.70 for this session until final billing posts.
Prior sessions have posted charges totaling $0.4020837270654738. Total charges plus this reserve are approximately $5.10 of $15.
The provider's partial posted amount for this session was $4.1067960490472615 at cleanup.
Do not add that partial amount to the reserve: the reserve already covers this session.
The post-termination account read returned no Pods. The local replacement timer was cancelled.
