"""Explicit NumPy oracle and unfused PyTorch reference, not a CUDA backend."""
import numpy as np
from .pipeline import Geometry, Conv2d


def numpy_stencil(source, kernel):
    """Flipped convolution, replicate borders, row-major separate FP32 multiply/add."""
    radius = len(kernel) // 2
    h, w, _ = source.shape
    result = np.zeros(source.shape, dtype=np.float32)
    for y, row in enumerate(kernel):
        iy = np.clip(np.arange(h) + radius - y, 0, h - 1)
        for x, coefficient in enumerate(row):
            ix = np.clip(np.arange(w) + radius - x, 0, w - 1)
            product = source[iy[:, None], ix[None, :]] * np.float32(coefficient)
            result = result + product
    return result


def _input(frame, pipeline):
    # Never coerce arbitrary objects: that could hide a device-to-host copy.
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("reference input must be a NumPy uint8 HWC array")
    plan = pipeline.plan(frame.shape)
    return plan, Geometry(**plan["geometry"])


def numpy_reference(pipeline, frame):
    """Bilinear half-pixel sampling, edge replication, no antialias, FP32 math."""
    plan, g = _input(frame, pipeline)
    stencil = pipeline.operations[0] if type(pipeline.operations[0]) is Conv2d else None
    size, norm, output = pipeline.operations[-3:]
    source = frame[..., ::-1] if pipeline.input_encoding == "bgr8" else frame
    source = source.astype(np.float32)
    if stencil is not None:
        source = numpy_stencil(source, stencil.kernel)
    y = np.maximum((np.arange(g.resized_height, dtype=np.float64)+.5)*g.input_height/g.resized_height-.5, 0)
    x = np.maximum((np.arange(g.resized_width, dtype=np.float64)+.5)*g.input_width/g.resized_width-.5, 0)
    y0, x0 = y.astype(np.int64), x.astype(np.int64)
    y1, x1 = np.minimum(y0+1, g.input_height-1), np.minimum(x0+1, g.input_width-1)
    wy, wx = (y-y0).astype(np.float32)[:, None, None], (x-x0).astype(np.float32)[None, :, None]
    upper = source[y0[:, None], x0[None, :]]*(1-wx) + source[y0[:, None], x1[None, :]]*wx
    lower = source[y1[:, None], x0[None, :]]*(1-wx) + source[y1[:, None], x1[None, :]]*wx
    canvas = np.full((g.output_height, g.output_width, 3), size.value, dtype=np.float32)
    canvas[g.top:g.top+g.resized_height, g.left:g.left+g.resized_width] = upper*(1-wy)+lower*wy
    canvas = (canvas*np.float32(norm.scale)-np.asarray(norm.mean, dtype=np.float32))/np.asarray(norm.std, dtype=np.float32)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=output.dtype), g


def torch_reference(pipeline, frame, *, device="cpu"):
    """Explicit upload from a CPU fixture; only CPU and MPS reference execution."""
    import torch
    import torch.nn.functional as F
    _, g = _input(frame, pipeline)
    if device not in ("cpu", "mps"):
        raise ValueError("reference device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; select cpu explicitly")
    stencil = pipeline.operations[0] if type(pipeline.operations[0]) is Conv2d else None
    size, norm, output = pipeline.operations[-3:]
    source = frame[..., ::-1] if pipeline.input_encoding == "bgr8" else frame
    tensor = torch.from_numpy(np.array(source, copy=True, order="C")).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)[None]
    if stencil is not None:
        coefficients = torch.tensor(stencil.kernel, dtype=torch.float32, device=device).flip((0, 1))
        n = len(stencil.kernel)
        tensor = F.conv2d(F.pad(tensor, (n//2,) * 4, mode='replicate'),
                          coefficients.view(1, 1, n, n).repeat(3, 1, 1, 1), groups=3)
    tensor = F.interpolate(tensor, size=(g.resized_height, g.resized_width), mode="bilinear", align_corners=False, antialias=False)
    tensor = F.pad(tensor, (g.left, g.output_width-g.resized_width-g.left,
                            g.top, g.output_height-g.resized_height-g.top), value=size.value)
    mean = torch.tensor(norm.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    tensor = (tensor*norm.scale-mean)/std
    return tensor.to(dtype=getattr(torch, output.dtype)).contiguous(), g
