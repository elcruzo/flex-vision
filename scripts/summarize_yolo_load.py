"""Recompute bounded detector load coverage, latency, and fairness from raw rows."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np


def summarize(root):
    report=json.loads((root/'results.json').read_text())
    if report['status']!='passed' or report['dirty']:raise ValueError('Require a clean passed load run')
    if len(report['runs']) != 12:raise ValueError('Require two loads, two repeats, and three candidates')
    seen=set();results=[]
    for run in report['runs']:
        identity=(run['fps_per_camera'],run['repeat'],run['candidate'])
        if identity in seen or identity[0] not in (60,400) or identity[1] not in (0,1) or identity[2] not in ('cpg','torch','cvcuda'):
            raise ValueError('Duplicate or unexpected load block')
        seen.add(identity)
        if run['status']!='passed' or run['failed_frames'] or run['seconds']!=30 or run['cameras']!=4 or run['depth']!=2:
            raise ValueError('Incomplete declared load protocol')
        path=root/run['samples']
        if not path.exists():path=Path(str(path)+'.gz')
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',newline='') as file:rows=list(csv.DictReader(file))
        total=4*identity[0]*30
        if len(rows)!=total or [int(r['frame']) for r in rows]!=list(range(total)):
            raise ValueError('Missing, duplicate, or reordered arrival records')
        completed=[]
        for row in rows:
            index=int(row['frame']);lane=int(row['lane'])
            if lane!=(index+identity[1])%4 or int(row['variant'])!=(index//4)%2:
                raise ValueError('Incorrect camera or fixture identity')
            arrival=float(row['arrival_s'])
            if abs(arrival-index/(4*identity[0]))>1e-8:raise ValueError('Incorrect arrival schedule')
            if row['status']=='dropped_queue_full':continue
            if row['status']!='completed' or row['correct']!='True':raise ValueError('Unchecked or failed completed frame')
            submit,done,gpu,latency=[float(row[k]) for k in ('submission_s','completion_s','gpu_pipeline_ms','arrival_to_completion_ms')]
            if not np.isfinite([submit,done,gpu,latency]).all() or not arrival<=submit<=done or gpu<=0 or latency<=0:
                raise ValueError('Invalid completed timing')
            if abs(latency-1000*(done-arrival))>1e-6:raise ValueError('Inconsistent completion boundary')
            if done>run['elapsed_with_drain_s']+.001:raise ValueError('Completion exceeds drain boundary')
            completed.append(row)
        if run['offered']!=total or run['completed']!=len(completed) or run['dropped']!=total-len(completed) or run['max_pending']>8:
            raise ValueError('Report counts differ from raw arrivals')
        for key,before in run['memory_before'].items():
            growth=run['memory_after'][key]-before
            if growth!=run['memory_growth'][key] or growth>report['allocation_growth_allowance_bytes']:
                raise ValueError('Allocator growth exceeds declared allowance')
        if not completed or not run['memory_samples']:raise ValueError('Missing correctness or memory evidence')
        lanes={}
        for lane in range(4):
            accepted=[r for r in completed if int(r['lane'])==lane]
            if not accepted:raise ValueError('Camera starved')
            lanes[str(lane)]={'completed':len(accepted),'offered':total//4,
                             'p99_ms':float(np.percentile([float(r['arrival_to_completion_ms']) for r in accepted],99))}
        latency=[float(r['arrival_to_completion_ms']) for r in completed]
        early=[float(r['arrival_to_completion_ms']) for r in completed if float(r['arrival_s'])<10]
        late=[float(r['arrival_to_completion_ms']) for r in completed if float(r['arrival_s'])>=20]
        results.append({'fps_per_camera':identity[0],'repeat':identity[1],'candidate':identity[2],
                        'offered':total,'completed':len(completed),'dropped':total-len(completed),
                        'completed_per_second':len(completed)/max(30,run['elapsed_with_drain_s']),
                        'arrival_to_completion_ms':dict(zip(('p50','p95','p99'),np.percentile(latency,[50,95,99]).tolist())),
                        'early_p99_ms':float(np.percentile(early,99)),'late_p99_ms':float(np.percentile(late,99)),
                        'memory_growth':run['memory_growth'],'lanes':lanes})
    return {'status':'verified','runs':results,'scope':report['scope'],
            'qualification':'30-second blocks with validation overhead; not concurrent streams, live cameras, or a soak'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.root)
    with args.output.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
