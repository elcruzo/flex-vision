"""Inspect a detector plan without importing PyTorch or allocating frame buffers."""
import json
from cpg import Pipeline

pipeline = (Pipeline(input_encoding="bgr8")
            .letterbox(640, 640, value=114)
            .normalize([0, 0, 0], [1, 1, 1], scale=1/255)
            .to(dtype="float16", layout="NCHW"))
print(json.dumps(pipeline.plan((1080, 1920, 3)), indent=2))
