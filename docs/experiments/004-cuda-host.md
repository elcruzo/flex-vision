# First CUDA host experiment

Result: passed hardware, interoperability, and real detector smoke checks.
This does not complete G0/G1 or prove the performance hypothesis.

Tested runner commit: `beeefa164160ce036c96c1c2cb3f1fe878ceff60`.
The traced detector run reported a clean working tree.

## Environment

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, compute capability 8.9 |
| Host driver | 580.178.04 |
| CUDA compiler | 13.0.88 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu130 |
| TorchVision | 0.24.1+cu130 |
| CuPy | 14.2.0, cupy-cuda13x |
| NumPy | 2.2.6 |
| Nsight Systems | 2025.3.2.474 |
| Container | runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster |
| Catalog source | Official template a9dk3g7cny |
| Host region | EU-RO-1, Secure Cloud |

The image tag is recorded. Its immutable registry digest was not collected.
The package inventory is retained with the local artifacts.

## Complete feedback loop

1. Load each pinned photograph and its padded variant.
2. Compute the independent NumPy letterbox reference.
3. Upload the BGR fixture to CuPy, then share its allocation with PyTorch through DLPack.
4. Execute unfused channel conversion, resize, padding, normalization, and layout operations on CUDA.
5. Feed the candidate tensor directly into pretrained SSDLite on CUDA.
6. Feed the independent reference into the same CUDA model.
7. Download evidence after inference. Compare tensors, dense heads, detections, and source-space object boxes.

All four cases passed in both the initial run and the committed traced run.
The maximum input tensor error was `1.531839370727539e-05`, below the fixed `2e-4` absolute tolerance.
Dense heads used absolute tolerance `0.01` and relative tolerance `1e-4`.
High-confidence labels matched. Scores and boxes passed tolerances `0.001` and `0.1` respectively.

| Photograph case | Expected-object source-box IoU |
| --- | --- |
| astronaut | 0.930434 |
| astronaut-padded | 0.863174 |
| chelsea | 0.944010 |
| chelsea-padded | 0.916463 |

The semantic checks require the expected class at confidence at least 0.5 and box IoU at least 0.5.
These are smoke annotations, not a representative accuracy dataset.

A separate same-stream check verified pointer identity through Torch → CuPy → Torch DLPack exchange.
A CuPy RawKernel compiled through NVRTC and modified the shared allocation. The resulting values matched exactly.
This establishes one interoperability path. It does not establish all ownership or cross-stream behavior.

## Trace evidence

The arithmetic preflight trace contains three CUDA kernel events and one 4096-byte device-to-host transfer.
The exported SQLite tables confirmed the exact transfer size and kernel count.
The detector trace includes preprocessing and inference NVTX ranges, CUDA activity, and deliberate validation transfers.
The entire harness includes model initialization, both candidate and baseline inference, and validation.
Its aggregate transfer counts are not production pipeline copy counts.

All 20 exported artifacts matched SHA256 values calculated on the remote host before termination.
Local artifact directory: `benchmark-results/iteration-004-runpod/benchmark-results/`.
The large traces and tensor outputs remain outside Git.

| Artifact | SHA256 |
| --- | --- |
| preflight-trace.nsys-rep | 9d284c22610a5ecf533252c8b6071f2fb2e24c487677dbc6088e9c2126ed9941 |
| detector-trace.nsys-rep | fa651fbd3332260a456fcaaf5deac8d8566dac2e6e50f8e32a1fc92924227932 |
| gpu-detector-committed/report.json | 14802022d731489be07c2e1c1cc5643a44189edb481d7c1bf900db5272117ba7 |

## Reproduce the environment

Use the recorded image on a compatible NVIDIA host. Create a separate environment that inherits its pinned CUDA PyTorch installation.
The template's system Python rejects direct package installation. Do not bypass that restriction.

```bash
export PATH=/usr/local/cuda/bin:$PATH
python -m venv --system-site-packages .venv-gpu
.venv-gpu/bin/python -m pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv-gpu/bin/python -m pip install -e . numpy==2.2.6 cupy-cuda13x==14.2.0
.venv-gpu/bin/python -m pip check
```

The framework pair follows [PyTorch's version instructions](https://pytorch.org/get-started/previous-versions/).
Download the pinned model with the instructions in [local development](../LOCAL_DEVELOPMENT.md).
The smoke runner checks the complete weight hash before loading it.

Install `nsight-systems-2025.3.2` from the image's configured NVIDIA apt repository.
Then execute:

```bash
.venv-gpu/bin/python scripts/gpu_preflight.py --output benchmark-results/preflight.json
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --output=benchmark-results/detector-trace .venv-gpu/bin/python scripts/gpu_detector_smoke.py --output benchmark-results/gpu-detector
nsys stats --report cuda_gpu_mem_size_sum,cuda_gpu_kern_sum,nvtx_sum --format csv benchmark-results/detector-trace.nsys-rep
```

Use fresh output paths. Preserve reports and trace files before deleting the host.
The fixture configuration is deliberately limited to the existing FP32 SSDLite contract.

CuPy deprecates `ExternalStream`, but this pinned PyTorch stream lacks `__cuda_stream__`.
The newer `Stream.from_external` call failed explicitly in this environment.
The runner retains the compatibility API and stream owner through completion.
See [CuPy interoperability](https://docs.cupy.dev/en/stable/user_guide/interoperability.html).

## Cost and cleanup

The user authorized $15 total, including later development sessions.
Pod `9rffjln920g60i` ran from 08:00:16 UTC until approximately 08:08:14 UTC on 2026-10-04.
The live catalog and created Pod reported $0.74/hour. Temporary disk was 50 GB.
Estimated compute plus disk cost is approximately $0.10. Final provider billing remains pending.
The billing query returned no posted records at cleanup. That does not mean the session was free.

A local termination guard and a Pod-side timer were configured as fallback controls.
These depend on process and API availability, so they are not provider-enforced spending caps.
The Pod was manually terminated after verified export, well before either timer expired.
The subsequent account read returned no Pods. No persistent or network volume was created.

## Next evidence required

- A CPG CUDA execution backend with the defined numerical contract.
- TensorRT consumption without intermediate host copies.
- CUDA Array Interface, cross-stream ownership, retained output, batching, and FP16 tests.
- GPU baseline comparisons and latency distributions under sustained load.
- Jetson, ROS transport, and representative camera footage.

The experiment supports further development. It does not support a speedup claim or production GPU-residency claim.
