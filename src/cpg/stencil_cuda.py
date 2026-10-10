"""Experimental coefficient-specialized CUDA stencil. Hardware acceptance pending."""
from functools import lru_cache
from .pipeline import Conv2d


def source(kernel):
    """Generate separate FP32 multiply/add statements in coefficient row order."""
    kernel = Conv2d(kernel).kernel
    radius = len(kernel) // 2
    statements = []
    for y, row in enumerate(kernel):
        for x, coefficient in enumerate(row):
            iy = f'min(max((long long)y+({radius-y}),0LL),(long long)h-1)'
            ix = f'min(max((long long)x+({radius-x}),0LL),(long long)w-1)'
            statements.append(f'    value = value + ((float)src[({iy})*sy+({ix})*sx+c*sc])*({coefficient:.9e}f);')
    return '''extern "C" __global__ void stencil(
    const unsigned char* src, float* dst, long long sy, long long sx, long long sc, int h, int w) {
    long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= (long long)h*w*3) return;
    int c = i%3, x = (i/3)%w, y = i/((long long)w*3);
    float value = 0.0f;
''' + '\n'.join(statements) + '\n    dst[i] = value;\n}\n'


@lru_cache(maxsize=32)
def kernel(coefficients):
    import cupy as cp
    return cp.RawKernel(source(coefficients), 'stencil', options=('--fmad=false',))
