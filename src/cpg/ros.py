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
