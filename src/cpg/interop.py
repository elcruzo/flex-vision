"""CUDA Array Interface metadata and explicit producer readiness."""
import numpy as np


def validate_cuda_interface(interface):
    """Snapshot the supported uint8 HWC version-3 contract without pixel access."""
    if not isinstance(interface, dict) or interface.get('version') != 3:
        raise TypeError('CUDA Array Interface version 3 is required')
    shape = interface.get('shape')
    if (not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError('CUDA Array Interface input must have positive HWC shape with three channels')
    if np.dtype(interface.get('typestr')) != np.dtype('uint8'):
        raise TypeError('CUDA Array Interface input must have dtype uint8')
    data = interface.get('data')
    if (not isinstance(data, tuple) or len(data) != 2
            or type(data[0]) is not int or data[0] <= 0 or type(data[1]) is not bool):
        raise ValueError('CUDA Array Interface requires a nonzero pointer and readonly flag')
    if interface.get('mask') is not None:
        raise ValueError('Masked CUDA Array Interface input is unsupported')
    strides = interface.get('strides')
    if strides is not None and (not isinstance(strides, tuple) or len(strides) != 3
                               or any(type(s) is not int for s in strides)):
        raise ValueError('CUDA Array Interface strides must be three integer byte strides')
    stream = interface.get('stream')
    if stream is not None and (type(stream) is not int or stream <= 0):
        raise ValueError('CUDA Array Interface stream must be None or a positive handle; zero is ambiguous')
    return dict(version=3, shape=shape, typestr='|u1', data=data, strides=strides, stream=stream)


class _View:
    def __init__(self, owner, interface):
        self.owner = owner
        self.__cuda_array_interface__ = dict(interface, stream=None)


class _StreamOwner:
    def __init__(self, owner, pointer):
        self.owner, self.pointer = owner, pointer

    def __cuda_stream__(self):
        return (0, self.pointer)


def import_cuda_interface(frame, cp):
    """Borrow pixels, retain the exporter, and queue its producer dependency."""
    interface = validate_cuda_interface(frame.__cuda_array_interface__)
    view = _View(frame, interface)
    source = cp.asarray(view)
    if source.device.id != cp.cuda.runtime.getDevice():
        raise ValueError('input must be on the current CUDA device')
    if (source.data.ptr != interface['data'][0] or tuple(source.shape) != interface['shape']
            or source.dtype != cp.uint8):
        raise ValueError('CUDA Array Interface import must preserve pointer, shape, and dtype')
    expected_strides = interface['strides'] or (interface['shape'][1] * 3, 3, 1)
    if tuple(source.strides) != expected_strides:
        raise ValueError('CUDA Array Interface import must preserve byte strides')
    owners = [frame, view]
    producer = None
    pointer = interface['stream']
    if pointer is not None:
        if pointer == 1:
            producer = cp.cuda.Stream.null
        elif pointer == 2:
            producer = cp.cuda.Stream.ptds
        else:
            producer = cp.cuda.Stream.from_external(_StreamOwner(frame, pointer))
        ready = cp.cuda.Event(disable_timing=True)
        ready.record(producer)
        cp.cuda.get_current_stream().wait_event(ready)
        owners.extend((producer, ready))
    return source, tuple(owners), producer
