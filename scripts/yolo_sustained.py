"""Serial resident-input YOLO load with bounded admission and checked inference."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import cupy as cp
import cvcuda
import numpy as np
import torch
from cpg.reference import numpy_reference
from yolo_common import pipeline,cases,decode,digest,LIMITS
from yolo_gpu import Consumer,torch_preprocess,vendor_preprocess
from yolo_load import Arrivals,FIELDS


def memory():
    pool = cp.get_default_memory_pool()
    return {'torch_allocated':torch.cuda.memory_allocated(),
            'torch_reserved':torch.cuda.memory_reserved(),
            'cupy_used':pool.used_bytes(),'cupy_total':pool.total_bytes()}


def valid(tensor,head,detection,expected,reference,reference_detection,candidate):
    # These two 1080p fixtures have exact tensors for all three measured paths.
    pixel = (torch.from_dlpack(tensor).view(torch.int16) == expected.view(torch.int16)).all()
    a,b = head.float(),reference.float()
    score_tol = .003 if candidate == 'cpg' else .03
    score = ((a[:,4:]-b[:,4:]).abs() <= score_tol+score_tol*b[:,4:].abs()).all()
    relevant = torch.ones((1,8400),device='cuda',dtype=torch.bool) if candidate == 'cpg' else torch.maximum(a[:,4:].amax(1),b[:,4:].amax(1)) >= .01
    boxes = (((a[:,:4]-b[:,:4]).abs() <= 4+.01*b[:,:4].abs()) | ~relevant[:,None,:]).all()
    sane = torch.isfinite(a).all() & (a[:,2:4]>=0).all() & (a[:,4:]>=0).all() & (a[:,4:]<=1).all()
    if any(detection[k].shape != reference_detection[k].shape for k in detection):
        return pixel & torch.zeros((),device='cuda',dtype=torch.bool)
    # Shared fixture controls have one confident detection. Ordered checks are
    # intentionally stronger than matching for this fixed repeated input.
    det = (detection['labels']==reference_detection['labels']).all()
    det &= ((detection['scores']-reference_detection['scores']).abs() <= .03).all()
    x,y = detection['boxes'],reference_detection['boxes']
    intersection = (torch.minimum(x[:,2:],y[:,2:])-torch.maximum(x[:,:2],y[:,:2])).clamp(min=0).prod(1)
    union = (x[:,2:]-x[:,:2]).prod(1)+(y[:,2:]-y[:,:2]).prod(1)-intersection
    det &= (intersection/union >= .95).all()
    return pixel & score & boxes & sane & det


def run(candidate,runner,consumer,sources,expected,references,detections,seconds,fps,depth,repeat):
    for i in range(100):
        variant=i%2
        tensor = runner(variant);head=consumer(tensor);result=decode(head)
        if not valid(tensor,head,result,expected[variant],references[variant],detections[variant],candidate).item():
            raise AssertionError('Checked warmup failed')
    del tensor,head,result
    torch.cuda.synchronize()
    before=memory()
    arrivals=Arrivals(4,fps,seconds,depth,repeat)
    events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    samples=[]
    start=time.perf_counter(); next_memory=0.; next_progress=10.
    while not arrivals.finished or time.perf_counter()-start < seconds:
        elapsed=time.perf_counter()-start
        if elapsed > seconds+30:
            raise TimeoutError('Arrival schedule did not drain within 30 seconds')
        arrivals.advance(elapsed)
        row=arrivals.pop()
        if row is None:
            time.sleep(.0001)
            continue
        row['submission_s']=time.perf_counter()-start
        variant=row['variant']
        events[0].record()
        tensor=runner(variant);head=consumer(tensor);result=decode(head)
        events[1].record()
        check=valid(tensor,head,result,expected[variant],references[variant],detections[variant],candidate)
        correct=bool(check.item())
        torch.cuda.current_stream().synchronize()
        row.update(status='completed',completion_s=time.perf_counter()-start,correct=correct,
                   gpu_pipeline_ms=events[0].elapsed_time(events[1]))
        row['arrival_to_completion_ms']=1000*(row['completion_s']-row['arrival_s'])
        del tensor,head,result,check
        if elapsed >= next_memory:
            samples.append({'elapsed_s':time.perf_counter()-start,**memory()});next_memory=elapsed+.5
        if elapsed >= next_progress:
            print(json.dumps({'event':'load_progress','candidate':candidate,'fps':fps,'repeat':repeat,
                              'elapsed_s':elapsed,'offered':arrivals.next}),flush=True)
            next_progress=elapsed+10
    elapsed=time.perf_counter()-start
    torch.cuda.synchronize()
    after=memory()
    completed=[r for r in arrivals.rows if r['status']=='completed']
    failed=[r['frame'] for r in completed if not r['correct']]
    growth={key:after[key]-before[key] for key in before}
    return {'candidate':candidate,'repeat':repeat,'fps_per_camera':fps,'cameras':4,'seconds':seconds,
            'depth':depth,'offered':arrivals.total,'completed':len(completed),
            'dropped':arrivals.total-len(completed),'max_pending':arrivals.max_pending,
            'elapsed_with_drain_s':elapsed,'failed_frames':failed,'memory_before':before,'memory_after':after,
            'memory_growth':growth,'memory_samples':samples,
            'status':'passed' if completed and not failed and all(v<=1048576 for v in growth.values()) else 'failed'},arrivals.rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=30)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--diagnose-vendor-aux',action='store_true')
    parser.add_argument('--soak',action='store_true',help='Thermal conditioning, then continuous 1800-second CPG detector run')
    args=parser.parse_args()
    if not 1<=args.seconds<=60 or not 1<=args.repeats<=4:parser.error('Invalid bounded run settings')
    if args.soak and (args.diagnose_vendor_aux or args.seconds!=30 or args.repeats!=2):
        parser.error('Soak uses a fixed protocol without short-matrix overrides')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'failed','revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
            'engine_sha256':digest(args.engine),'gpu':torch.cuda.get_device_name(),
            'scope':'four logical cameras, one persistent serial stream; resident synthetic inputs; GPU checks inside offered load',
            'allocation_growth_allowance_bytes':1048576,'runs':[]}
    try:
        if report['dirty']:raise ValueError('Commit measured source first')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        consumer=Consumer(args.engine);pipe=pipeline();cvstream=cvcuda.Stream()
        stream=torch.cuda.ExternalStream(cvstream.handle)
        with torch.inference_mode(),torch.cuda.stream(stream),cp.cuda.ExternalStream(stream.cuda_stream,device_id=0):
            sources=[];expected=[];references=[];detections=[];geometries=[]
            for name,rgb,_,_ in cases():
                if not name.endswith('1080p'):continue
                tensor,g=numpy_reference(pipe,np.ascontiguousarray(rgb[...,::-1]))
                control=np.load(args.controls/(name+'-controls.npz'))
                np.testing.assert_array_equal(tensor,control['tensor'])
                sources.append(cp.asarray(np.ascontiguousarray(rgb[...,::-1])))
                expected.append(torch.from_numpy(tensor).cuda());geometries.append(g)
                reference=consumer(cp.from_dlpack(expected[-1]))
                torch.testing.assert_close(reference,torch.from_numpy(control['tensorrt_dense']).cuda(),atol=0,rtol=0)
                references.append(reference);detections.append(decode(reference))
            if len(sources)!=2:raise ValueError('Require both 1080p controls')
            runners={'cpg':lambda v:pipe(sources[v]),
                     'torch':lambda v:torch_preprocess(sources[v],geometries[v]),
                     'cvcuda':lambda v:vendor_preprocess(sources[v],geometries[v],cvstream)}
            # Prime all candidates before allocator baselines. Keep the same stream.
            for candidate,runner in runners.items():
                for v in range(2):
                    tensor=runner(v);head=consumer(tensor);d=decode(head)
                    if not valid(tensor,head,d,expected[v],references[v],detections[v],candidate).item():
                        raise AssertionError('Initial load control failed')
            del tensor,head,d
            if args.soak:
                from yolo_soak import run_soak
                report['soak']=run_soak(run,runners['cpg'],consumer,sources,expected,references,detections,args.output)
                report['status']=report['soak']['status']
                if report['status']!='passed':raise AssertionError('Detector soak failed')
                return
            for fps in (60,400):
                for repeat in range(args.repeats):
                    names=list(runners);names=names[repeat%3:]+names[:repeat%3]
                    for candidate in names:
                        result,rows=run(candidate,runners[candidate],consumer,sources,expected,references,detections,args.seconds,fps,2,repeat)
                        filename=f'{fps}-{repeat}-{candidate}.csv'
                        with (args.output/filename).open('w',newline='') as file:
                            writer=csv.DictWriter(file,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
                        result['samples']=filename;report['runs'].append(result)
                        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            if args.diagnose_vendor_aux:
                # Version-pinned private API: diagnostic only, never a baseline
                # strategy and never a production dependency.
                sync_aux=getattr(getattr(cvcuda,'internal',None),'syncAuxStream',None)
                if sync_aux is None:
                    report['resource_diagnostic']={'status':'unsupported','reason':'Pinned auxiliary synchronization hook unavailable'}
                else:
                    def diagnostic_runner(variant):
                        output=vendor_preprocess(sources[variant],geometries[variant],cvstream)
                        sync_aux()
                        return output
                    diagnostic,rows=run('cvcuda_aux_diagnostic',diagnostic_runner,consumer,sources,expected,references,detections,args.seconds,400,2,0)
                    with (args.output/'aux-diagnostic.csv').open('w',newline='') as file:
                        writer=csv.DictWriter(file,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
                    diagnostic['samples']='aux-diagnostic.csv'
                    diagnostic['qualification']='Private auxiliary-stream hook; resource-lifetime diagnostic only, excluded from performance comparison'
                    report['resource_diagnostic']=diagnostic
        report['status']='passed' if all(run['status']=='passed' for run in report['runs']) else 'failed'
        if report['status']!='passed':raise AssertionError('Completed all load blocks; correctness or allocation growth failed')
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
