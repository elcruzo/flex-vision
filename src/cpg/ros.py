"""Explicit CPU standard-Image views. No ROS import, CUDA bridge, or upload."""
from array import array
from dataclasses import dataclass, field
import numpy as np


def _uint32(value, name, *, positive=False):
    if type(value) is not int or not (int(positive) <= value < 2**32):
        raise ValueError(f'{name} must be a valid {"positive " if positive else ""}uint32')
    return value


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    step: int
    encoding: str
    frame_id: str
    stamp_sec: int
    stamp_nanosec: int


@dataclass(frozen=True)
class CPUImageView:
    pixels: np.ndarray = field(repr=False)
    metadata: ImageMetadata
    owner: object = field(repr=False)


def cpu_image_view(message, *, expected_encoding):
    """Borrow validated host bytes while retaining the ROS message owner.

    Only bytes, bytearray, and unsigned-byte array payloads are accepted.
    Arbitrary buffers and GPU-backed objects are rejected without coercion.
    The caller must keep the underlying message unmodified while reading.
    """
    if expected_encoding not in ('rgb8','bgr8'):
        raise ValueError('Expected encoding must be rgb8 or bgr8')
    width=_uint32(message.width,'width',positive=True)
    height=_uint32(message.height,'height',positive=True)
    step=_uint32(message.step,'step',positive=True)
    if message.encoding != expected_encoding:
        raise ValueError('Image encoding does not match the pipeline input contract')
    if type(message.is_bigendian) is not int or message.is_bigendian not in (0,1):
        raise ValueError('Invalid image byte-order field')
    # Eight-bit channels are unaffected by the declared byte order.
    if step < width*3:
        raise ValueError('Image row stride is smaller than its RGB/BGR payload')
    data=message.data
    if type(data) not in (bytes,bytearray,array) or (type(data) is array and data.typecode!='B'):
        raise TypeError('Require explicit host byte storage; native CUDA buffers need a separate bridge')
    if len(data) != height*step:
        raise ValueError('Image byte count must equal height times row stride')
    header=message.header
    sec,ns=header.stamp.sec,header.stamp.nanosec
    if type(sec) is not int or not -2**31 <= sec < 2**31 or type(ns) is not int or not 0 <= ns < 1_000_000_000:
        raise ValueError('Invalid Image timestamp')
    if not isinstance(header.frame_id,str):
        raise ValueError('Image frame_id must be a string')
    pixels=np.ndarray((height,width,3),dtype=np.uint8,buffer=data,strides=(step,3,1))
    pixels.setflags(write=False)
    return CPUImageView(pixels,ImageMetadata(width,height,step,message.encoding,header.frame_id,sec,ns),message)


@dataclass(frozen=True)
class CPUTensorView:
    pixels: np.ndarray = field(repr=False)
    owner: object = field(repr=False)


def write_cpu_tensor(message, pixels):
    """Copy an explicit contiguous host FP16/FP32 tensor into a ROS message.

    This CPU serialization path copies payload bytes. It is not a GPU bridge.
    Layout and camera metadata require a separate consumer contract.
    """
    if type(pixels) is not np.ndarray or pixels.dtype not in (np.dtype('float16'), np.dtype('float32')):
        raise TypeError('Require a NumPy FP16 or FP32 tensor')
    if not pixels.flags.c_contiguous or not pixels.shape or any(d <= 0 for d in pixels.shape):
        raise ValueError('Require a nonempty contiguous tensor')
    message.dtype_code = 2
    message.dtype_bits = pixels.dtype.itemsize * 8
    message.dtype_lanes = 1
    message.shape = list(pixels.shape)
    message.strides = []
    message.byte_offset = 0
    message.data = array('B', pixels.tobytes())
    return message


def cpu_tensor_view(message, *, expected_shape, expected_dtype):
    """Borrow a strict contiguous standard tensor's explicit host storage."""
    dtype = np.dtype(expected_dtype)
    if dtype not in (np.dtype('float16'), np.dtype('float32')):
        raise ValueError('Expected tensor dtype must be FP16 or FP32')
    shape = tuple(message.shape)
    if not shape or any(type(d) is not int or d <= 0 for d in shape) or shape != tuple(expected_shape):
        raise ValueError('Tensor shape does not match the consumer contract')
    fields = (message.dtype_code, message.dtype_bits, message.dtype_lanes)
    if any(type(value) is not int for value in fields) or fields != (2, dtype.itemsize * 8, 1):
        raise ValueError('Tensor dtype does not match the consumer contract')
    if len(message.strides) or type(message.byte_offset) is not int or message.byte_offset != 0:
        raise ValueError('Require contiguous tensor storage with zero byte offset')
    data = message.data
    if type(data) not in (bytes, bytearray, array) or (type(data) is array and data.typecode != 'B'):
        raise TypeError('Require explicit host tensor storage; CUDA storage needs a native bridge')
    import math
    if len(data) != math.prod(shape) * dtype.itemsize:
        raise ValueError('Tensor byte count does not match its shape and dtype')
    pixels = np.ndarray(shape, dtype=dtype, buffer=data)
    pixels.setflags(write=False)
    return CPUTensorView(pixels, message)


def write_tensor_geometry(message, geometry):
    """Write actual letterbox geometry into the experimental stamped envelope."""
    message.source_width = geometry.input_width
    message.source_height = geometry.input_height
    message.resized_width = geometry.resized_width
    message.resized_height = geometry.resized_height
    message.padding_left = geometry.left
    message.padding_top = geometry.top
    # Validate the completed envelope against the tensor dimensions.
    tensor_geometry(message)
    return message


def tensor_geometry(message):
    """Validate stamped NCHW geometry before mapping model boxes to source."""
    from .pipeline import Geometry
    if message.layout != 'NCHW' or message.source_encoding not in ('rgb8', 'bgr8'):
        raise ValueError('Require RGB/BGR source and NCHW model layout')
    shape = tuple(message.tensor.shape)
    if len(shape) != 4 or shape[:2] != (1, 3) or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError('Require one three-channel NCHW model tensor')
    width = _uint32(message.source_width, 'source_width', positive=True)
    height = _uint32(message.source_height, 'source_height', positive=True)
    resized_width = _uint32(message.resized_width, 'resized_width', positive=True)
    resized_height = _uint32(message.resized_height, 'resized_height', positive=True)
    left = _uint32(message.padding_left, 'padding_left')
    top = _uint32(message.padding_top, 'padding_top')
    if left + resized_width > shape[3] or top + resized_height > shape[2]:
        raise ValueError('Resized image and padding exceed the model tensor')
    return Geometry(height, width, resized_height, resized_width, shape[2], shape[3], top, left)
