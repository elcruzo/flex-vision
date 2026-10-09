"""First fused CUDA backend. Calls synchronize their stream before returning."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=2)
def _kernel(dtype):
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
    return cp.RawKernel(source, 'preprocess', options=('--fmad=false',))


def execute(pipeline, frame):
    """Return a CuPy NCHW allocation without host pixel copies.

    Accept CuPy or CUDA DLPack input on the current device. CuPy callers must
    order their producer on the current stream. DLPack performs its stream
    handoff. Completion is synchronous to make input lifetime explicit.
    """
    if not hasattr(frame, '__dlpack_device__') or frame.__dlpack_device__()[0] != 2:
        raise TypeError('input must be a CUDA DLPack array; CPU uploads must be explicit')
    import cupy as cp
    if frame.__dlpack_device__()[1] != cp.cuda.runtime.getDevice():
        raise ValueError('input must be on the current CUDA device')
    source = frame if isinstance(frame, cp.ndarray) else cp.from_dlpack(frame)
    if source.dtype != cp.uint8:
        raise TypeError('input must have dtype uint8')
    shape, dtype, count, constants = _launch_metadata(pipeline, tuple(int(n) for n in source.shape))
    result = cp.empty(shape, dtype=dtype)
    args = (source, result, *(np.int64(s) for s in source.strides), *constants)
    stream = cp.cuda.get_current_stream()
    _kernel(dtype)(((count+255)//256,), (256,), args, stream=stream)
    # Keep source, imported owner, and output alive until their work completes.
    stream.synchronize()
    return result


@lru_cache(maxsize=64)
def _launch_metadata(pipeline, shape):
    """Cache immutable host metadata only. Never retain arrays or stream owners."""
    plan = pipeline.plan(shape, backend='cuda')
    g = plan['geometry']
    size, norm, output = pipeline.operations
    dims = [g[k] for k in ('input_height', 'input_width', 'output_height', 'output_width',
                           'resized_height', 'resized_width', 'top', 'left')]
    if any(n > np.iinfo(np.int32).max for n in dims):
        raise ValueError('image dimensions exceed the CUDA backend limit')
    constants = (*(np.int32(n) for n in dims), np.int32(pipeline.input_encoding == 'bgr8'),
                 *(np.float32(v) for v in (size.value, norm.scale, *norm.mean, *norm.std)))
    output_shape = tuple(plan['output_shape'])
    return output_shape, output.dtype, int(np.prod(output_shape)), constants
