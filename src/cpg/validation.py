"""Local fixture and detection checks; not part of GPU execution."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def box_iou(a, b):
    """Intersection-over-union for continuous xyxy boxes."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        raise ValueError("boxes must contain four finite coordinates")
    if (a[2:] < a[:2]).any() or (b[2:] < b[:2]).any():
        raise ValueError("boxes must be ordered xyxy")
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = (a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod() - intersection
    return float(intersection/union) if union > 0 else 0.


def check_expected_object(labels, scores, source_boxes, expectation):
    """Require a semantic detection, not merely agreement between two empty outputs."""
    if (type(expectation["label"]) is not int or expectation["label"] <= 0
            or not np.isfinite(expectation["min_score"]) or not 0 <= expectation["min_score"] <= 1
            or not np.isfinite(expectation["min_iou"]) or not 0 < expectation["min_iou"] <= 1):
        raise ValueError("expectation needs a positive label and valid finite score/IoU thresholds")
    box_iou(expectation["box_xyxy"], expectation["box_xyxy"])
    if not (len(labels) == len(scores) == len(source_boxes)):
        raise ValueError("labels, scores, and boxes must have matching lengths")
    matches = [box_iou(box, expectation["box_xyxy"])
               for label, score, box in zip(labels, scores, source_boxes)
               if int(label) == expectation["label"] and float(score) >= expectation["min_score"]]
    best = max(matches, default=0.)
    if best < expectation["min_iou"]:
        raise AssertionError(f"Expected {expectation['name']} with score >= {expectation['min_score']} "
                             f"and source-box IoU >= {expectation['min_iou']}; best IoU={best:.4f}")
    return best


def photo_cases(manifest_path):
    """Verify shipped image bytes before decode and yield original/padded fixtures."""
    from PIL import Image
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    seen = set()
    for entry in manifest["images"]:
        if not isinstance(entry["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["id"]):
            raise ValueError("Fixture ID must contain only letters, digits, underscores, and hyphens")
        if entry["id"] in seen or entry["id"]+"-padded" in seen:
            raise ValueError("Fixture IDs must be unique, including derived padded IDs")
        seen.update((entry["id"], entry["id"]+"-padded"))
        path = manifest_path.parent / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Fixture checksum mismatch: {path}")
        with Image.open(path) as image:
            frame = np.array(image.convert("RGB"), dtype=np.uint8)
        if list(frame.shape) != entry["shape"]:
            raise ValueError(f"Fixture shape mismatch: {path}")
        yield entry["id"], frame, entry["expectation"], {"file": entry["file"], "sha256": entry["sha256"], "transform": "none"}
        # A known placement checks source-box mapping with non-square, odd-sized images.
        height, width, _ = frame.shape
        padded = np.full((height+61, width+103, 3), 114, dtype=np.uint8)
        padded[17:17+height, 29:29+width] = frame
        expected = dict(entry["expectation"])
        expected["box_xyxy"] = (np.asarray(expected["box_xyxy"])+[29, 17, 29, 17]).tolist()
        yield entry["id"]+"-padded", padded, expected, {"file": entry["file"], "sha256": entry["sha256"],
                                                     "transform": "pad bottom/right 44/74; top/left 17/29; value 114"}
