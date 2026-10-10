"""Fixed specialized-stencil tensors through actual TensorRT inference."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output already exists')
    import cupy as cp
    import numpy as np
    import torch
    from cpg.pipeline import Conv2d
    from cpg.reference import numpy_reference
    from yolo_common import pipeline,cases,digest,decode,compare_dense,compare_detections
    from pending_yolo_consumer import PendingConsumer
    pre,downstream=[cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    consumer=PendingConsumer(args.engine,downstream)
    report=dict(status='failed',mode='stencil-inference-v1',revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),engine_sha256=digest(args.engine),checks=[],
                scope='Six fixtures, four sizes, positive/negative asymmetric stencils, contiguous/reversed channels, sync/submit, FP16 TensorRT. No performance or full stream acceptance.')
    pending=job=None
    retained=None
    try:
        if report['dirty']:raise ValueError('Require clean committed source')
        base=pipeline()
        with torch.inference_mode():
            for name,rgb,_,_ in cases():
                host=np.ascontiguousarray(rgb[...,::-1])
                for size in (3,5,7,9):
                    for variant in ('positive','negative'):
                        coefficients=[[0.]*size for _ in range(size)]
                        edge,center=(.125,.75) if variant=='positive' else (-.25,1.5)
                        coefficients[0][0]=edge;coefficients[size//2][size//2]=center;coefficients[-1][0]=edge
                        pipe=replace(base,operations=(Conv2d(tuple(tuple(r) for r in coefficients)),)+base.operations)
                        expected,_=numpy_reference(pipe,host)
                        with pre:oracle=cp.asarray(expected)
                        pre.synchronize()
                        reference=consumer(oracle)
                        with torch.cuda.stream(consumer._torch_stream):reference_detection=decode(reference)
                        downstream.synchronize()
                        for layout in ('contiguous','reverse_channels'):
                            for execution in ('sync','submit'):
                                report['active_scenario']=dict(fixture=name,size=size,variant=variant,layout=layout,execution=execution)
                                with pre:
                                    frame=cp.asarray(host if layout=='contiguous' else np.ascontiguousarray(host[...,::-1]))
                                    if layout=='reverse_channels':frame=frame[...,::-1]
                                    if execution=='sync':image=pipe(frame)
                                    else:
                                        pending=pipe.submit(frame,stream=pre)
                                        image=pending.wait_on(downstream)
                                del frame
                                if execution=='sync':downstream.wait_event(pre.record())
                                job=consumer.enqueue(image)
                                head=job.wait();job=None
                                if pending is not None:pending.close();pending=None
                                np.testing.assert_array_equal(cp.asnumpy(image).view(np.uint16),expected.view(np.uint16))
                                if retained is not None:
                                    np.testing.assert_array_equal(cp.asnumpy(retained[0]).view(np.uint16),retained[1].view(np.uint16))
                                with torch.cuda.stream(consumer._torch_stream):
                                    compare_dense(reference,head,strict_output=True)
                                    compare_detections(reference_detection,decode(head))
                                downstream.synchronize()
                                report['checks'].append(dict(fixture=name,size=size,variant=variant,layout=layout,execution=execution,tensor_exact=True,
                                    dense_output=True,detections=True,retained_output_exact=True if retained is not None else None,
                                    plan=pipe.plan(host.shape,backend='cuda')))
                                retained=(image,expected)
                                del image,head
                        del oracle,reference,reference_detection
                print('stencil_validated',name,flush=True)
        report['active_scenario']=None;report['status']='passed'
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        try:
            if pending is not None:pending.close()
            if job is not None:job.close()
            pre.synchronize();downstream.synchronize();report['drain_status']='passed'
        except Exception as exc:
            report['status']='failed';report['drain_status']='failed';report['drain_error']=repr(exc);raise
        finally:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
