"""Static explanations for the current fixed detector execution strategies."""


def explain(pipeline, shape, *, backend='reference'):
    """Describe execution boundaries without importing CUDA or measuring a frame."""
    plan = pipeline.plan(shape, backend=backend)
    operations = plan['operations']
    has_stencil = operations[0]['kind'] == 'Conv2d'
    output_bytes = 1
    for dimension in plan['output_shape']:
        output_bytes *= dimension
    output_bytes *= 2 if plan['dtype'] == 'float16' else 4
    groups = []
    if backend == 'cuda':
        if has_stencil:
            groups.append(dict(operations=['Conv2d'], implementation='specialized-direct-experimental',
                               input='borrowed uint8 HWC', output='owned FP32 HWC temporary',
                               cuda_launches=1))
        groups.append(dict(operations=[op['kind'] for op in operations[int(has_stencil):]],
                           implementation='internal-fused-detector',
                           input='owned FP32 HWC temporary' if has_stencil else 'borrowed uint8 HWC',
                           output='owned inference tensor or optional synchronous caller destination',
                           cuda_launches=1))
    else:
        groups = [dict(operations=[op['kind']], implementation='explicit-reference',cuda_launches=None)
                  for op in operations]
    return dict(plan=plan, execution_groups=groups,
                memory=dict(output_bytes=output_bytes,
                            temporary_pixel_bytes=shape[0]*shape[1]*3*4 if has_stencil and backend=='cuda' else 0 if backend=='cuda' else None,
                            qualification='Pixel buffers only. Excludes allocator pools, compiled modules, input ownership, and inference memory.'),
                measurements=dict(status='unmeasured',p50_ms=None,p95_ms=None,p99_ms=None,
                                  host_to_device_bytes=None,device_to_host_bytes=None,peak_temporary_bytes=None),
                qualification='Static fixed-graph explanation. No execution, tuning, timing, transfer verification, or CUDA hardware acceptance.')
