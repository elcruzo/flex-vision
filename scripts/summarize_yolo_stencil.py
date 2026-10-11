"""Verify complete stencil timing records and compute metrics from raw samples."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def summarize(root):
    import numpy as np
    report = json.loads((root/'results.json').read_text())
    if report.get('status') != 'passed' or report.get('drain_status') != 'passed' or report.get('dirty') is not False:
        raise ValueError('Require passed, drained, clean-source experiment')
    if report.get('mode') != 'stencil-comparison-v1' or not re.fullmatch('[0-9a-f]{40}',report.get('revision','')):
        raise ValueError('Unexpected source or protocol')
    if not re.fullmatch('[0-9a-f]{64}',report.get('engine_sha256','')):
        raise ValueError('Require engine identity')
    from verify_yolo_stencil import verify
    acceptance_bytes = (root/'acceptance.json').read_bytes()
    if hashlib.sha256(acceptance_bytes).hexdigest() != report.get('acceptance_sha256'):
        raise ValueError('Acceptance report hash mismatch')
    accepted = verify(json.loads(acceptance_bytes))
    if any(accepted[field] != report[field] for field in ('revision','engine_sha256')):
        raise ValueError('Acceptance source or engine mismatch')
    blocks,samples = report['blocks'],report['samples']
    if type(blocks) is not int or blocks < 2 or blocks%2 or type(samples) is not int or samples < 100:
        raise ValueError('Invalid sampling protocol')
    checks = report['checks']
    identities = [(x['fixture'],x['size'],x['path']) for x in checks]
    fixtures = sorted({x[0] for x in identities})
    expected = {(fixture,size,path) for fixture in fixtures for size in (3,5,7,9) for path in ('cpg','pytorch')}
    if len(fixtures) != 2 or len(identities) != 16 or set(identities) != expected:
        raise ValueError('Incomplete or duplicated numerical acceptance')
    if any(x.get('inference') != 'passed' or not math.isfinite(x['tensor_max_error']) or x['tensor_max_error'] < 0
           or x['tensor_max_error'] > (0 if x['path']=='cpg' else report['limits']['tensor_atol']) for x in checks):
        raise ValueError('Numerical acceptance failed')
    groups = {key:[] for key in expected}
    seen = set()
    fields = ('preprocess_ms','inference_ms','decode_ms','gpu_total_ms','host_ms')
    with (root/'timings.csv').open(newline='') as file:
        for row in csv.DictReader(file):
            key = (row['fixture'],int(row['size']),row['path'])
            block,sample = int(row['block']),int(row['sample'])
            identity = (*key,block,sample)
            if key not in expected or not 0 <= block < blocks or not 0 <= sample < samples or identity in seen:
                raise ValueError('Invalid or duplicated timing identity')
            values = [float(row[field]) for field in fields]
            if any(not math.isfinite(x) or x < 0 for x in values) or values[-1] <= 0:
                raise ValueError('Invalid timing value')
            seen.add(identity)
            groups[key].append(values)
    if len(seen) != 16*blocks*samples:
        raise ValueError('Incomplete timing coverage')
    summaries = []
    for (fixture,size,path),values in sorted(groups.items()):
        array = np.array(values)
        summaries.append(dict(fixture=fixture,size=size,path=path,count=len(values),
                             metrics={field:dict(zip(('p50','p95','p99'),map(float,np.percentile(array[:,i],[50,95,99]))))
                                      for i,field in enumerate(fields)}))
    return dict(status='coverage-verified',revision=report['revision'],engine_sha256=report['engine_sha256'],
                completed_inference_checks=len(seen),samples=summaries,
                qualification='Derived from exported timings and declared per-frame checks; no independent GPU replay, transfer verification, sustained throughput, or fastest-backend claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():raise ValueError('Summary already exists')
    args.output.write_text(json.dumps(summarize(args.root),indent=2)+'\n')


if __name__ == '__main__':main()
