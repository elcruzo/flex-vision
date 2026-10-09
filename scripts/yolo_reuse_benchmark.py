"""Matched fresh versus caller-owned output through checked TensorRT inference."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import cupy as cp
import numpy as np
import torch
from cpg.reference import numpy_reference
from yolo_common import pipeline,cases,decode,digest
from yolo_gpu import Consumer
from yolo_sustained import valid


class AllocationCounter(cp.cuda.MemoryHook):
    name='CPGAllocationCounter'
    def __init__(self):
        self.pool_requests=0;self.requested_bytes=0;self.device_allocations=0;self.device_bytes=0
    def malloc_preprocess(self,**kwargs):
        self.pool_requests+=1;self.requested_bytes+=kwargs['size']
    def alloc_preprocess(self,**kwargs):
        self.device_allocations+=1;self.device_bytes+=kwargs['mem_size']
    def record(self):return {key:getattr(self,key) for key in ('pool_requests','requested_bytes','device_allocations','device_bytes')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'failed','revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
            'engine_sha256':digest(args.engine),'gpu':torch.cuda.get_device_name(),
            'blocks':10,'samples_per_block':1000,'warmup':100,'fixtures':[],
            'scope':'serial synchronous resident 1080p input; fixed FP16 TensorRT and CUDA NMS; validation after timing; no live capture',
            'timing_boundary':'preprocess host/GPU and complete host/GPU stop before per-frame GPU validation; validation may condition subsequent samples',
            'allocation_scope':'separate CuPy hooks around preprocessing only; pool requests differ from actual device allocations',
            'common_memory_state':'one caller-owned slot remains allocated for both candidates; no pool clearing',
            'acceptance':'all sampled frames correct; classify p99 host regression above 5% on either fixture before adoption; no minimum speedup assumed'}
    try:
        if report['dirty']:raise ValueError('Commit measured source first')
        pipe=pipeline();consumer=Consumer(args.engine);stream=torch.cuda.Stream()
        with torch.inference_mode(),torch.cuda.stream(stream),cp.cuda.ExternalStream(stream.cuda_stream,device_id=0):
            slot=cp.empty((1,3,640,640),dtype=cp.float16)
            events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            with (args.output/'samples.csv').open('w',newline='') as handle:
                writer=csv.writer(handle)
                writer.writerow(['fixture','block','sample','candidate','preprocess_host_ms','preprocess_gpu_ms','complete_host_ms','complete_gpu_ms','correct'])
                for name,rgb,_,_ in cases():
                    if not name.endswith('1080p'):continue
                    host=np.ascontiguousarray(rgb[...,::-1]);source=cp.asarray(host)
                    tensor,_=numpy_reference(pipe,host)
                    control=np.load(args.controls/(name+'-controls.npz'))
                    np.testing.assert_array_equal(tensor,control['tensor'])
                    expected=torch.from_numpy(tensor).cuda();reference=consumer(cp.from_dlpack(expected));detection=decode(reference)
                    torch.testing.assert_close(reference,torch.from_numpy(control['tensorrt_dense']).cuda(),atol=0,rtol=0)
                    runners={'fresh':lambda:pipe(source),'reuse':lambda:pipe(source,out=slot)}
                    for runner in runners.values():
                        for _ in range(100):
                            result=runner();head=consumer(result);decoded=decode(head)
                            if not valid(result,head,decoded,expected,reference,detection,'cpg').item():raise AssertionError('Warmup failed')
                            del result,head,decoded
                    allocation={}
                    for candidate,runner in runners.items():
                        hook=AllocationCounter()
                        for _ in range(100):
                            with hook:result=runner()
                            head=consumer(result);decoded=decode(head)
                            if not valid(result,head,decoded,expected,reference,detection,'cpg').item():raise AssertionError('Allocation control failed')
                            del result,head,decoded
                        allocation[candidate]={'frames':100,**hook.record()}
                    record={'name':name,'allocations':allocation,'checked_frames':0,'failed_frames':0};report['fixtures'].append(record)
                    for block in range(10):
                        order=('fresh','reuse') if block%2==0 else ('reuse','fresh')
                        for candidate in order:
                            for index in range(1000):
                                start=time.perf_counter_ns();events[0].record()
                                result=runners[candidate]();preprocess_host=(time.perf_counter_ns()-start)/1e6;events[1].record()
                                head=consumer(result);decoded=decode(head);events[2].record();stream.synchronize()
                                complete_host=(time.perf_counter_ns()-start)/1e6
                                # The backend synchronized preprocessing, so this
                                # event records its end before inference starts.
                                complete_gpu=events[0].elapsed_time(events[2])
                                correct=bool(valid(result,head,decoded,expected,reference,detection,'cpg').item())
                                writer.writerow([name,block,index,candidate,preprocess_host,events[0].elapsed_time(events[1]),complete_host,complete_gpu,correct])
                                record['checked_frames']+=1;record['failed_frames']+=not correct
                                del result,head,decoded
                        handle.flush();print('reuse_timed',name,block,flush=True)
            if len(report['fixtures'])!=2 or any(x['failed_frames'] for x in report['fixtures']):raise AssertionError('Matched reuse correctness failed')
            report['status']='passed'
    except Exception as exc:report['error']=repr(exc);raise
    finally:(args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
