"""Immutable detector graph, independent of CUDA and tensor libraries."""
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real


def _dimension(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    value: float

    def __post_init__(self):
        _dimension(self.width, "width")
        _dimension(self.height, "height")
        if not 0 <= _finite(self.value, "padding value") <= 255:
            raise ValueError("padding value must be in [0, 255]")


@dataclass(frozen=True)
class Normalize:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    scale: float

    def __post_init__(self):
        if type(self.mean) is not tuple or type(self.std) is not tuple:
            raise ValueError("mean/std must be immutable tuples")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean/std must contain three channels")
        for value in self.mean:
            _finite(value, "mean")
        if any(_finite(value, "std") <= 0 for value in self.std):
            raise ValueError("std must be positive")
        _finite(self.scale, "scale")


@dataclass(frozen=True)
class Convert:
    dtype: str
    layout: str

    def __post_init__(self):
        if self.dtype not in ("float16", "float32") or self.layout != "NCHW":
            raise ValueError("output must be float16 or float32 with layout NCHW")


@dataclass(frozen=True)
class Geometry:
    input_height: int
    input_width: int
    resized_height: int
    resized_width: int
    output_height: int
    output_width: int
    top: int
    left: int

    @property
    def scale_xy(self):
        return (self.resized_width / self.input_width,
                self.resized_height / self.input_height)

    def source_box(self, box):
        """Map continuous xyxy edge coordinates to the source, clipping padding."""
        if len(box) != 4:
            raise ValueError("box must contain four xyxy coordinates")
        x1, y1, x2, y2 = (_finite(v, "box coordinate") for v in box)
        if x2 < x1 or y2 < y1:
            raise ValueError("box coordinates must be ordered")
        sx, sy = self.scale_xy
        return (min(self.input_width, max(0., (x1-self.left)/sx)),
                min(self.input_height, max(0., (y1-self.top)/sy)),
                min(self.input_width, max(0., (x2-self.left)/sx)),
                min(self.input_height, max(0., (y2-self.top)/sy)))


@dataclass(frozen=True)
class Pipeline:
    """Initial grammar: letterbox -> normalize -> to. Always outputs RGB."""
    operations: tuple = ()
    input_encoding: str = "rgb8"

    def __post_init__(self):
        if self.input_encoding not in ("rgb8", "bgr8"):
            raise ValueError("input_encoding must be rgb8 or bgr8")
        if not isinstance(self.operations, tuple):
            raise ValueError("operations must be an immutable tuple")
        expected = (Letterbox, Normalize, Convert)
        if len(self.operations) > 3 or any(type(op) is not expected[i] for i, op in enumerate(self.operations)):
            raise ValueError("supported order is letterbox -> normalize -> to")

    def letterbox(self, width, height, value=114):
        value = _finite(value, "padding value")
        return replace(self, operations=self.operations + (Letterbox(width, height, value),))

    def normalize(self, mean, std, *, scale):
        mean = tuple(_finite(v, "mean") for v in mean)
        std = tuple(_finite(v, "std") for v in std)
        return replace(self, operations=self.operations + (Normalize(mean, std, _finite(scale, "scale")),))

    def to(self, *, dtype, layout):
        return replace(self, operations=self.operations + (Convert(dtype, layout),))

    def plan(self, shape, *, backend='reference'):
        if backend not in ('reference', 'cuda'):
            raise ValueError('backend must be reference or cuda')
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("input shape must be HWC with three channels")
        h, w, _ = shape
        _dimension(h, "input height")
        _dimension(w, "input width")
        if len(self.operations) != 3:
            raise ValueError("complete letterbox -> normalize -> to before planning")
        size, _, output = self.operations
        ratio = min(size.width / w, size.height / h)
        # Round half up and preserve at least one pixel on each axis.
        rw = min(size.width, max(1, math.floor(w * ratio + 0.5)))
        rh = min(size.height, max(1, math.floor(h * ratio + 0.5)))
        geometry = Geometry(h, w, rh, rw, size.height, size.width,
                            (size.height-rh)//2, (size.width-rw)//2)
        return {"backend": "cuda-fused" if backend == 'cuda' else "reference-only", "input_encoding": self.input_encoding,
                "output_encoding": "rgb", "output_shape": [1, 3, size.height, size.width],
                "dtype": output.dtype, "geometry": asdict(geometry),
                "operations": [{"kind": type(op).__name__, **asdict(op)} for op in self.operations],
                "cuda_launches": 1 if backend == 'cuda' else None, "fusion": backend == 'cuda',
                "synchronous": backend == 'cuda',
                "temporary_arrays": 0 if backend == 'cuda' else None}

    def __call__(self, frame, *, out=None):
        from .cuda import execute
        return execute(self, frame, out=out)
