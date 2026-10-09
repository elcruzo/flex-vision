"""Independently check continuous detector soak rows and telemetry."""
import argparse
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import numpy as np


def summarize(root):
    report=json.loads((root/'results.json').read_text())
    soak=report['soak'];run=soak['run']
    if report['dirty'] or report['status']!='passed' or soak['status']!='passed':
        raise ValueError('Require clean passed soak evidence')
    if (run['candidate'],run['seconds'],run['cameras'],run['fps_per_camera'],run['depth'])!=('cpg',1800,4,60,2):
        raise ValueError('Incorrect continuous soak protocol')
    if run['status']!='passed' or run['failed_frames'] or run['max_pending']>8:
        raise ValueError('Failed soak checks or queue bounds')
    if not np.isfinite(run['elapsed_with_drain_s']) or not 1800<=run['elapsed_with_drain_s']<=1830:raise ValueError('Invalid soak duration')
    path=root/run['samples']
    if not path.exists():path=Path(str(path)+'.gz')
    opener=gzip.open if path.suffix=='.gz' else open
    bins=[[] for _ in range(6)];lanes=[0]*4;completed=0;offered=0
    with opener(path,'rt',newline='') as handle:
        for index,row in enumerate(csv.DictReader(handle)):
            offered+=1;arrival=index/240
            if not np.isfinite(float(row['arrival_s'])):raise ValueError('Invalid arrival timestamp')
            if int(row['frame'])!=index or int(row['lane'])!=index%4 or int(row['variant'])!=(index//4)%2 or abs(float(row['arrival_s'])-arrival)>1e-8:
                raise ValueError('Incomplete or incorrect arrival identity')
            if row['status']=='dropped_queue_full':continue
            if row['status']!='completed' or row['correct']!='True':raise ValueError('Unchecked completed frame')
            submit,done,gpu,latency=[float(row[k]) for k in ('submission_s','completion_s','gpu_pipeline_ms','arrival_to_completion_ms')]
            if not np.isfinite([submit,done,gpu,latency]).all() or not arrival<=submit<=done<=run['elapsed_with_drain_s']+.001 or gpu<=0 or latency<=0 or abs(latency-1000*(done-arrival))>1e-6:
                raise ValueError('Invalid completion timing')
            bins[min(5,int(arrival//300))].append(latency)
            completed+=1;lanes[index%4]+=1
    if offered!=432000 or run['offered']!=offered or run['completed']!=completed or run['dropped']!=offered-completed or not all(lanes) or not all(bins):
        raise ValueError('Counts or coverage disagree')
    samples=run['memory_samples']
    sample_times=[x['elapsed_s'] for x in samples]
    if not sample_times or not np.isfinite(sample_times).all() or sample_times[0]<0 or sample_times[0]>2 or sample_times[-1]<1798 or any(not 0<b-a<=2 for a,b in zip(sample_times,sample_times[1:])):
        raise ValueError('Incomplete allocator sampling')
    expected_keys={'torch_allocated','torch_reserved','cupy_used','cupy_total'}
    if any(set(run[k])!=expected_keys for k in ('memory_before','memory_after','memory_growth')) or set(soak['sampled_memory_ceiling_growth'])!=expected_keys:
        raise ValueError('Missing allocator counters')
    counter_values=[v for k in ('memory_before','memory_after','memory_growth') for v in run[k].values()]
    counter_values+=list(soak['sampled_memory_ceiling_growth'].values())
    if not np.isfinite(counter_values).all():raise ValueError('Invalid allocator values')
    if set(run['memory_before'])!=expected_keys or any(set(x)-{'elapsed_s'}!=expected_keys for x in samples):
        raise ValueError('Missing allocator counters')
    if not np.isfinite([x[k] for x in samples for k in expected_keys]).all():
        raise ValueError('Invalid allocator samples')
    for key,before in run['memory_before'].items():
        growth=run['memory_after'][key]-before
        if growth!=run['memory_growth'][key] or growth>1048576:raise ValueError('Post-drain allocator growth failed')
        early=[x[key] for x in run['memory_samples'] if x['elapsed_s']<300]
        late=[x[key] for x in run['memory_samples'] if 1500<=x['elapsed_s']<1800]
        if not early or not late:raise ValueError('Missing memory windows')
        growth=max(late)-max(early)
        if growth!=soak['sampled_memory_ceiling_growth'][key] or growth>1048576:raise ValueError('Sampled allocator growth failed')
    telemetry=list(csv.reader((root/'telemetry.csv').read_text().splitlines()))
    start=soak['telemetry_start_row']
    if not isinstance(start,int) or start<120 or start>=len(telemetry):raise ValueError('Invalid telemetry boundary')
    conditioning=telemetry[start-120:start]
    if len(conditioning)!=120:raise ValueError('Missing conditioning telemetry')
    if any(len(x)!=6 for x in telemetry):raise ValueError('Invalid GPU telemetry columns')
    if not np.isfinite([float(x) for row in telemetry for x in row[1:]]).all():raise ValueError('Invalid GPU telemetry values')
    temperatures=[float(x[1]) for x in conditioning]
    if max(temperatures)-min(temperatures)>2 or abs(np.median(temperatures[60:])-np.median(temperatures[:60]))>1:
        raise ValueError('Thermal stabilization failed')
    rows=telemetry[start:]
    times=[datetime.strptime(x[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp() for x in rows]
    if len(rows)<1790 or times[-1]-times[0]<1790 or any(not 0<b-a<=3 for a,b in zip(times,times[1:])):
        raise ValueError('Incomplete or interrupted soak telemetry')
    for row in rows:
        if len(row)!=6 or not np.isfinite([float(x) for x in row[1:]]).all():raise ValueError('Invalid GPU telemetry')
    return {'status':'verified','offered':offered,'completed':completed,'dropped':offered-completed,
            'per_feed_completed':lanes,'five_minute_latency_ms':[dict(zip(('p50','p95','p99'),np.percentile(x,[50,95,99]).tolist())) for x in bins],
            'memory_growth':run['memory_growth'],'sampled_memory_ceiling_growth':soak['sampled_memory_ceiling_growth'],
            'telemetry_rows':len(rows),'qualification':'CPG serial synthetic resident detector soak with correctness overhead; no comparative speed or live-camera claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.root)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
