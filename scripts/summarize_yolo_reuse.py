"""Audit matched output-reuse samples and allocation observations."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np

FIELDS=('preprocess_host_ms','preprocess_gpu_ms','complete_host_ms','complete_gpu_ms')


def summarize(root):
    report=json.loads((root/'results.json').read_text())
    if report['status']!='passed' or report['dirty'] or report['blocks']!=10 or report['samples_per_block']!=1000 or report['warmup']!=100:
        raise ValueError('Require complete clean matched reuse experiment')
    fixtures={x['name']:x for x in report['fixtures']}
    if set(fixtures)!= {'astronaut-1080p','chelsea-1080p'} or len(report['fixtures'])!=2:raise ValueError('Incorrect fixture coverage')
    path=root/'samples.csv'
    if not path.exists():path=Path(str(path)+'.gz')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='') as handle:rows=list(csv.DictReader(handle))
    expected=[(name,block,sample,candidate) for name in fixtures for block in range(10)
              for candidate in (('fresh','reuse') if block%2==0 else ('reuse','fresh')) for sample in range(1000)]
    if len(rows)!=40000:raise ValueError('Missing matched samples')
    for row,identity in zip(rows,expected):
        if (row['fixture'],int(row['block']),int(row['sample']),row['candidate'])!=identity or row['correct']!='True':raise ValueError('Incorrect identity, order, or correctness')
        values=[float(row[k]) for k in FIELDS]
        if not np.isfinite(values).all() or min(values)<=0 or values[0]>values[2] or values[1]>values[3]:raise ValueError('Invalid timing boundaries')
    result={'status':'verified','samples':40000,'fixtures':{},'qualification':report['scope'],'adoption':'acceptable'}
    for name,fixture in fixtures.items():
        if fixture['checked_frames']!=20000 or fixture['failed_frames']:raise ValueError('Incorrect checked-frame coverage')
        timing={}
        for candidate in ('fresh','reuse'):
            selected=[r for r in rows if r['fixture']==name and r['candidate']==candidate]
            pooled={k:dict(zip(('p50','p95','p99'),np.percentile([float(r[k]) for r in selected],[50,95,99]).tolist())) for k in FIELDS}
            blocks=[{k:dict(zip(('p50','p95','p99'),np.percentile([float(r[k]) for r in selected if int(r['block'])==b],[50,95,99]).tolist())) for k in FIELDS} for b in range(10)]
            timing[candidate]={'pooled':pooled,'blocks':blocks}
            observed=fixture['allocations'][candidate]
            if observed['frames']!=100 or any(not isinstance(observed[k],int) or observed[k]<0 for k in ('pool_requests','requested_bytes','device_allocations','device_bytes')):
                raise ValueError('Incomplete allocation observations')
        # The hook must detect at least the owned output as its instrumentation control.
        fresh,reuse=fixture['allocations']['fresh'],fixture['allocations']['reuse']
        if fresh['pool_requests']<100 or fresh['requested_bytes']<100*2457600:raise ValueError('Allocation-hook positive control failed')
        regression=timing['reuse']['pooled']['complete_host_ms']['p99']/timing['fresh']['pooled']['complete_host_ms']['p99']-1
        if regression>.05:result['adoption']='rejected_p99_regression'
        result['fixtures'][name]={'timing':timing,'allocations':fixture['allocations'],
                                'complete_host_p99_regression_fraction':regression,
                                'preprocess_host_p50_gain_fraction':1-timing['reuse']['pooled']['preprocess_host_ms']['p50']/timing['fresh']['pooled']['preprocess_host_ms']['p50']}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=summarize(args.root)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({'status':result['status'],'samples':result['samples'],'adoption':result['adoption']}))


if __name__=='__main__':main()
