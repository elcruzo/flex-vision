"""Check caller-owned output reuse through actual TensorRT inference on CUDA."""
import argparse
import json
from pathlib import Path
import subprocess
import cupy as cp
import numpy as np
import torch
from cpg.reference import numpy_reference
from yolo_common import pipeline,cases,decode,compare_dense,compare_detections,digest
from yolo_gpu import Consumer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('Output report already exists')
    report={'status':'failed','revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
            'engine_sha256':digest(args.engine),'cases':[],
            'qualification':'Synchronous caller-owned CuPy outputs; validation downloads explicit; no async or allocation performance claim'}
    try:
        if report['dirty']:raise ValueError('Commit measured source first')
        pipe=pipeline();consumer=Consumer(args.engine)
        stream=torch.cuda.Stream()
        with torch.inference_mode(),torch.cuda.stream(stream),cp.cuda.ExternalStream(stream.cuda_stream,device_id=0):
            outputs=[cp.empty((1,3,640,640),dtype=cp.float16) for _ in range(2)]
            # A slot is reused only after inference and validation complete.
            retained=None
            for name,rgb,_,_ in cases():
                host=np.ascontiguousarray(rgb[...,::-1]);source=cp.asarray(host)
                expected,_=numpy_reference(pipe,host)
                controls=np.load(args.controls/(name+'-controls.npz'))
                np.testing.assert_array_equal(expected,controls['tensor'])
                reference=torch.from_numpy(controls['tensorrt_dense']).cuda()
                for index in range(4):
                    slot=outputs[index%2]
                    result=pipe(source,out=slot)
                    assert result is slot
                    np.testing.assert_array_equal(cp.asnumpy(result).view(np.uint16),expected.view(np.uint16))
                    head=consumer(result)
                    compare_dense(reference,head,strict_output=True)
                    compare_detections(decode(reference),decode(head))
                    if retained is not None and retained[0] is not slot:
                        np.testing.assert_array_equal(cp.asnumpy(retained[0]),retained[1])
                    stream.synchronize()
                    retained=(slot,cp.asnumpy(slot))
                # Default calls must still return distinct owned arrays.
                fresh=pipe(source);snapshot=cp.asnumpy(fresh)
                pipe(source,out=outputs[0])
                np.testing.assert_array_equal(cp.asnumpy(fresh),snapshot)
                assert all(fresh.data.ptr!=slot.data.ptr for slot in outputs)
                # Import a Torch producer through the existing DLPack dependency.
                result=pipe(torch.from_dlpack(source),out=outputs[1])
                np.testing.assert_array_equal(cp.asnumpy(result).view(np.uint16),expected.view(np.uint16))
                compare_detections(decode(reference),decode(consumer(result)))
                stream.synchronize()
                rejected=[]
                bad_outputs={'dtype':cp.empty((1,3,640,640),dtype=cp.float32),
                             'shape':cp.empty((1,3,1,1),dtype=cp.float16),
                             'layout':cp.empty((1,3,640,1280),dtype=cp.float16)[...,::2],
                             'type':torch.empty((1,3,640,640),device='cuda',dtype=torch.float16)}
                if source.nbytes>=outputs[0].nbytes:
                    bad_outputs['alias']=source.ravel()[:outputs[0].nbytes].view(cp.float16).reshape(outputs[0].shape)
                for reason,destination in bad_outputs.items():
                    try:pipe(source,out=destination)
                    except (TypeError,ValueError):rejected.append(reason)
                    else:raise AssertionError('Invalid caller-owned output accepted: '+reason)
                np.testing.assert_array_equal(cp.asnumpy(source),host)
                report['cases'].append({'name':name,'checked_reuses':4,'rejected_outputs':rejected,'torch_input':'passed',
                                        'default_output_retention':'passed','status':'passed'})
            report['status']='passed'
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
