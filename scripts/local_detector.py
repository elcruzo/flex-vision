"""Run local reference preprocessing through a real pretrained detector.

No CUDA or TensorRT claim. Model inference runs on CPU. MPS, when requested,
only executes the PyTorch preprocessing reference; its output is downloaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from cpg import load_pipeline
from cpg.reference import numpy_reference, torch_reference

WEIGHTS_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
WEIGHTS_URL = "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(height, width):
    y, x = np.indices((height, width))
    return np.stack(((x*255//max(width-1, 1)), (y*255//max(height-1, 1)),
                     ((x//32+y//32)%2)*255), axis=-1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="Preprocessing device only")
    parser.add_argument("--config", type=Path, default=Path("examples/local-detector.yaml"))
    parser.add_argument("--weights", type=Path, default=Path(".cache/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing runs are never overwritten")
    args = parser.parse_args()
    if not args.weights.is_file() or digest(args.weights) != WEIGHTS_SHA256:
        parser.error(f"Expected pinned weights SHA256 {WEIGHTS_SHA256}. Download from {WEIGHTS_URL}")
    pipeline = load_pipeline(args.config)
    plan = pipeline.plan((1080, 1920, 3))
    norm = pipeline.operations[1]
    if (pipeline.input_encoding != "bgr8" or plan["output_shape"] != [1, 3, 320, 320]
            or plan["dtype"] != "float32" or norm.mean != (0., 0., 0.)
            or norm.std != (1., 1., 1.) or norm.scale != 1/255):
        parser.error("This local SSDLite fixture requires bgr8 -> 320x320 RGB float32 NCHW, scale 1/255, mean 0, std 1")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
                "preprocessing_device": args.device, "inference_device": "cpu", "weights_sha256": WEIGHTS_SHA256,
                "config_sha256": digest(args.config), "config_text": args.config.read_text(),
                "weights_url": WEIGHTS_URL, "model": "ssdlite320_mobilenet_v3_large.COCO_V1",
                "scope": "synthetic local reference validation; not accuracy, CUDA, TensorRT, or performance evidence",
                "tensor_atol": 2e-4, "dense_head_atol": 0.01, "dense_head_rtol": 1e-4,
                "postprocess_box_atol": 0.1, "postprocess_score_atol": 0.001,
                "source_files": {str(p): digest(p) for p in sorted(Path("src/cpg").glob("*.py"))},
                "runner_sha256": digest(Path(__file__))}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    results = []
    status = "failed"
    try:
        torch.set_num_threads(4)
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
        # The detector keeps its own trained mean/std transform. External scale is 1/255 only.
        captures = []
        def capture(_module, _inputs, outputs):
            captures.append({key: value.detach().clone() for key, value in outputs.items()})
        hook = model.head.register_forward_hook(capture)
        for height, width in [(1080, 1920), (481, 639), (33, 17)]:
            frame = fixture(height, width)
            # Pass a non-contiguous BGR view to exercise explicit encoding and strides.
            bgr = frame[..., ::-1]
            case = {"input_shape": list(bgr.shape), "input_sha256": hashlib.sha256(bgr.tobytes()).hexdigest(),
                    "plan": pipeline.plan(bgr.shape), "status": "failed"}
            results.append(case)
            started = time.perf_counter()
            expected, geometry = numpy_reference(pipeline, bgr)
            actual, _ = torch_reference(pipeline, bgr, device=args.device)
            actual_cpu = actual.cpu()  # Deliberate validation transfer; never a residency claim.
            case["tensor_max_abs_error"] = float(np.abs(actual_cpu.numpy()-expected).max())
            np.testing.assert_allclose(actual_cpu.numpy(), expected, atol=manifest["tensor_atol"], rtol=0)
            captures.clear()
            with torch.inference_mode():
                baseline = model([torch.from_numpy(expected[0])])[0]
                candidate = model([actual_cpu[0]])[0]
            case["dense_head_errors"] = {}
            for key in captures[0]:
                case["dense_head_errors"][key] = float((captures[0][key]-captures[1][key]).abs().max())
                torch.testing.assert_close(captures[0][key], captures[1][key], atol=manifest["dense_head_atol"], rtol=manifest["dense_head_rtol"])
            # Ignore low confidence proposals for the postprocessing check, but compare all dense outputs above.
            detections = []
            for output in (baseline, candidate):
                keep = output["scores"] >= .5
                detections.append({k: v[keep] for k, v in output.items()})
            torch.testing.assert_close(detections[0]["labels"], detections[1]["labels"], atol=0, rtol=0)
            torch.testing.assert_close(detections[0]["scores"], detections[1]["scores"], atol=manifest["postprocess_score_atol"], rtol=0)
            torch.testing.assert_close(detections[0]["boxes"], detections[1]["boxes"], atol=manifest["postprocess_box_atol"], rtol=0)
            np.savez_compressed(args.output/f"outputs-{height}x{width}.npz",
                                expected_tensor=expected, actual_tensor=actual_cpu.numpy(),
                                baseline_boxes=baseline["boxes"].numpy(), candidate_boxes=candidate["boxes"].numpy(),
                                baseline_scores=baseline["scores"].numpy(), candidate_scores=candidate["scores"].numpy(),
                                baseline_labels=baseline["labels"].numpy(), candidate_labels=candidate["labels"].numpy())
            case["detections_above_0_5"] = len(detections[1]["labels"])
            case["source_boxes"] = [geometry.source_box(box.tolist()) for box in detections[1]["boxes"]]
            case["diagnostic_elapsed_seconds"] = time.perf_counter()-started
            case["status"] = "passed"
        hook.remove()
        status = "passed"
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output/"results.json").write_text(json.dumps({"status": status, "cases": results, "error": manifest.get("error")}, indent=2)+"\n")
        report = f"# Local detector reference run\n\nStatus: {status}.\n\nPreprocessing: {args.device}. Inference: CPU SSDLite.\n\n"
        report += "Synthetic fixtures only. No detection accuracy or GPU performance claim.\n"
        report += "Dense head outputs are compared even when no detection exceeds the score threshold.\n"
        report += "MPS output download, when selected, is deliberate and outside the future CUDA residency contract.\n"
        (args.output/"report.md").write_text(report)
    print(json.dumps({"status": status, "cases": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
