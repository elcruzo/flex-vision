"""Collect non-secret GPU setup evidence and execute a small CUDA stream check.

This is a smoke check, not CPG/TensorRT acceptance. It installs nothing.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess


def command(args):
    if not shutil.which(args[0]):
        return {"status": "missing"}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "error": str(exc)}


def collect():
    report = {"schema_version": 1, "platform": platform.system(), "architecture": platform.machine(),
              "python": platform.python_version(), "status": "blocked",
              "scope": "CUDA arithmetic/stream smoke only; not TensorRT inference, CPG, or tracing acceptance",
              "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
              "nvcc": command(["nvcc", "--version"]), "nsys": command(["nsys", "--version"]),
              "packages": {}, "pending": ["CuPy interoperability", "TensorRT inference", "actual Nsight trace inspection"]}
    for name in ("torch", "torchvision", "cupy-cuda13x", "cupy-cuda12x", "tensorrt", "tensorrt-cu13", "cuda-toolkit"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"]:
            report["reason"] = "No CUDA device available to this PyTorch environment"
            return report
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        # Independent stream work plus explicit completion before a known D2H transfer.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            values = torch.arange(1024, dtype=torch.float32, device="cuda:0")
            result = values * 2 + 1
            finished = torch.cuda.Event()
            finished.record(stream)
        finished.synchronize()
        torch.testing.assert_close(result.cpu(), torch.arange(1024, dtype=torch.float32)*2+1, rtol=0, atol=0)
        report["status"] = "cuda_smoke_passed"
        report["known_d2h_bytes"] = 4096
        report["reason"] = "CUDA arithmetic and explicit stream completion passed; inspect the separate trace next"
    except Exception as exc:
        report["status"] = "failed"
        report["reason"] = f"{type(exc).__name__}: {exc}"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output before GPU work; never overwrite earlier evidence.
    with args.output.open("x") as output:
        report = collect()
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "cuda_smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
