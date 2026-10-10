"""Fused CUDA backend with synchronous calls and experimental submissions."""
from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=4)
def _kernel(dtype, input_dtype='uint8'):
    import cupy as cp
    output_type = 'half' if dtype == 'float16' else 'float'
    cast = '__float2half_rn(value)' if dtype == 'float16' else 'value'
    source = r'''
#include <cuda_fp16.h>
extern "C" __global__ void preprocess(
    const unsigned char* src, OUT* dst,
    long long sy, long long sx, long long sc,
    int ih, int iw, int oh, int ow, int rh, int rw, int top, int left,
    int bgr, float pad, float scale,
    float m0, float m1, float m2, float s0, float s1, float s2) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long plane = (long long)oh * ow;
    if (i >= 3 * plane) return;
    int c = i / plane;
    int y = (i % plane) / ow, x = i % ow;
    float value = pad;
    if (y >= top && y < top + rh && x >= left && x < left + rw) {
        // Double coordinates match the independent oracle before FP32 weights.
        double fy = fmax(0.0, ((double)(y-top)+0.5)*ih/rh-0.5);
        double fx = fmax(0.0, ((double)(x-left)+0.5)*iw/rw-0.5);
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = min(y0+1, ih-1), x1 = min(x0+1, iw-1);
        float wy = (float)(fy-y0), wx = (float)(fx-x0);
        int channel = bgr ? 2-c : c;
        float a = src[y0*sy+x0*sx+channel*sc];
        float b = src[y0*sy+x1*sx+channel*sc];
        float d = src[y1*sy+x0*sx+channel*sc];
        float e = src[y1*sy+x1*sx+channel*sc];
        float upper = a*(1-wx)+b*wx, lower = d*(1-wx)+e*wx;
        value = upper*(1-wy)+lower*wy;
    }
    float mean = c == 0 ? m0 : (c == 1 ? m1 : m2);
    float std = c == 0 ? s0 : (c == 1 ? s1 : s2);
    value = (value*scale-mean)/std;
    dst[i] = CAST;
}
'''.replace('OUT', output_type).replace('CAST', cast)
    if input_dtype == 'float32':
        source = source.replace('const unsigned char* src', 'const float* src')
    elif input_dtype != 'uint8':
        raise ValueError('Unsupported preprocessing source dtype')
    return cp.RawKernel(source, 'preprocess', options=('--fmad=false',))


def execute(pipeline, frame, *, out=None):
    return _execute(pipeline, frame, out=out, asynchronous=False)


def submit(pipeline, frame, *, stream):
    """Experimental owned-output submission on an explicit CuPy stream."""
    import cupy as cp
    if not isinstance(stream, cp.cuda.Stream):
        raise TypeError("stream must be a CuPy Stream")
    if stream.device_id != cp.cuda.runtime.getDevice():
        raise ValueError("stream must be an explicit current-device stream")
    with stream:
        return _execute(pipeline, frame, out=None, asynchronous=True)


def _execute(pipeline, frame, *, out, asynchronous):
    """Return a CuPy NCHW allocation without host pixel copies.

    Accept CuPy, CUDA DLPack, or experimental version-3 CAI input. CuPy callers must
    order their producer on the current stream. DLPack performs its stream
    handoff. Calls synchronize; submissions retain owners until completion.
    """
    dlpack = getattr(frame, '__dlpack_device__', None)
    cai = dlpack is None and hasattr(frame, '__cuda_array_interface__')
    if dlpack is None and not cai:
        raise TypeError('input must be a CUDA DLPack or CUDA Array Interface array; CPU uploads must be explicit')
    if dlpack is not None and dlpack()[0] != 2:
        raise TypeError('input must be a CUDA DLPack array; CPU uploads must be explicit')
    import cupy as cp
    owners = frame
    producer = None
    if dlpack is not None:
        if dlpack()[1] != cp.cuda.runtime.getDevice():
            raise ValueError('input must be on the current CUDA device')
        source = frame if isinstance(frame, cp.ndarray) else cp.from_dlpack(frame)
    else:
        from .interop import import_cuda_interface
        source, owners, producer = import_cuda_interface(frame, cp)
    if source.dtype != cp.uint8:
        raise TypeError('input must have dtype uint8')
    shape, dtype, count, constants = _launch_metadata(pipeline, tuple(int(n) for n in source.shape))
    if out is None:
        result = cp.empty(shape, dtype=dtype)
    else:
        if not isinstance(out, cp.ndarray):
            raise TypeError('out must be a CuPy array')
        _validate_output(out, source, shape, dtype, cp.cuda.runtime.getDevice())
        result = out
    stream = cp.cuda.get_current_stream()
    # Create the completion owner before dispatch so errors cannot orphan a read.
    event = cp.cuda.Event(disable_timing=True) if asynchronous else None
    temporary = None
    try:
        from .pipeline import Conv2d
        stencil = pipeline.operations[0] if type(pipeline.operations[0]) is Conv2d else None
        launch_source = source
        if stencil is not None:
            from .stencil_cuda import kernel
            temporary = cp.empty(source.shape, dtype=cp.float32)
            stencil_count = int(source.size)
            stencil_args = (source, temporary, *(np.int64(s) for s in source.strides),
                            np.int32(source.shape[0]), np.int32(source.shape[1]))
            kernel(stencil.kernel)(((stencil_count+255)//256,), (256,), stencil_args, stream=stream)
            launch_source = temporary
        args = (launch_source, result, *(np.int64(s//launch_source.dtype.itemsize) for s in launch_source.strides), *constants)
        if stencil is None:
            _kernel(dtype)(((count+255)//256,), (256,), args, stream=stream)
        else:
            _kernel(dtype, 'float32')(((count+255)//256,), (256,), args, stream=stream)
        if asynchronous:
            event.record(stream)
            if producer is not None:
                producer.wait_event(event)
            return Submission(owners if temporary is None else (owners, temporary), source, result, stream, event, source.device.id)
        stream.synchronize()
        return result
    except BaseException:
        # Drain any submitted read before local owners leave this frame.
        stream.synchronize()
        raise


@lru_cache(maxsize=64)
def _launch_metadata(pipeline, shape):
    """Cache immutable host metadata only. Never retain arrays or stream owners."""
    plan = pipeline.plan(shape, backend='cuda')
    g = plan['geometry']
    size, norm, output = pipeline.operations[-3:]
    dims = [g[k] for k in ('input_height', 'input_width', 'output_height', 'output_width',
                           'resized_height', 'resized_width', 'top', 'left')]
    if any(n > np.iinfo(np.int32).max for n in dims):
        raise ValueError('image dimensions exceed the CUDA backend limit')
    constants = (*(np.int32(n) for n in dims), np.int32(pipeline.input_encoding == 'bgr8'),
                 *(np.float32(v) for v in (size.value, norm.scale, *norm.mean, *norm.std)))
    output_shape = tuple(plan['output_shape'])
    count = math.prod(output_shape)
    if count > np.iinfo(np.int64).max:
        raise ValueError('output element count exceeds the CUDA backend limit')
    return output_shape, output.dtype, count, constants


def _byte_interval(array):
    """Conservative byte extent, including negative input strides and gaps."""
    pointer = int(array.data.ptr)
    offsets = [(int(n)-1)*int(s) for n,s in zip(array.shape,array.strides)]
    return pointer+sum(min(0,x) for x in offsets), pointer+sum(max(0,x) for x in offsets)+array.dtype.itemsize


def _validate_output(out, source, shape, dtype, device):
    """Validate metadata before dispatch. Do not read pixel payloads."""
    if out.device.id != device:
        raise ValueError('out must be on the current CUDA device')
    if tuple(out.shape) != shape:
        raise ValueError(f'out must have shape {shape}')
    if out.dtype != np.dtype(dtype):
        raise TypeError(f'out must have dtype {dtype}')
    if not out.flags.c_contiguous:
        raise ValueError('out must be C-contiguous NCHW')
    if int(out.data.ptr) % out.dtype.itemsize:
        raise ValueError('out pointer must be aligned to its dtype')
    a,b = _byte_interval(source),_byte_interval(out)
    if a[0] < b[1] and b[0] < a[1]:
        raise ValueError('out must not overlap the input byte extent')


class Submission:
    """Experimental preprocessing completion. Does not track external consumers.

    Use wait() for host completion or wait_on(stream) for a consumer dependency.
    Retain the returned array until that consumer finishes. close() drains only
    preprocessing; it never authorizes overwriting a consumer's output.
    """
    def __init__(self, frame, source, output, stream, event, device):
        self._owners = (frame, source)
        self._output = output
        self._stream = stream
        self._event = event
        self._device = device
        self._completed = False

    def wait(self):
        import cupy as cp
        with cp.cuda.Device(self._device):
            if not self._completed:
                self._event.synchronize()
                self._completed = True
                self._owners = ()
        return self._output

    def wait_on(self, stream):
        import cupy as cp
        if not isinstance(stream, cp.cuda.Stream):
            raise TypeError('consumer stream must be a CuPy Stream')
        if stream.device_id != self._device:
            raise ValueError('consumer stream must be on the submission device')
        with cp.cuda.Device(self._device):
            stream.wait_event(self._event)
        return self._output

    def close(self):
        self.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # Conservative fallback only. Explicit close/context use is required.
        if hasattr(self, '_completed') and not self._completed:
            self.close()
