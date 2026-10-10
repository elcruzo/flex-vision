"""Independently verify matched YOLO sample coverage and recompute latency statistics."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

CANDIDATES = ('cpg','torch','cvcuda')
FIXTURES = ('astronaut-1080p','chelsea-1080p')
FIELDS = ('preprocess_ms','network_ms','nms_ms','complete_gpu_ms','complete_host_ms')


def summarize(root):
    report = json.loads((root/'results.json').read_text())
    if report.get('run_kind', 'comparison') != 'comparison':
        raise ValueError('Require a latency comparison, not validation or trace evidence')
    if report['status'] != 'passed' or report['dirty']:
        raise ValueError('Require a completed run from a clean source revision')
    if report['blocks'] < 10 or report['samples_per_block'] < 1000 or report['warmup'] < 100:
        raise ValueError('Insufficient declared repetitions')
    expected_cases = {'astronaut','astronaut-padded','chelsea','chelsea-padded',*FIXTURES}
    if {case['id'] for case in report['cases']} != expected_cases or len(report['cases']) != 6:
        raise ValueError('Incorrect correctness coverage')
    for case in report['cases']:
        if case['status'] != 'passed' or set(case['paths']) != {*CANDIDATES,'torch_dlpack','negative_stride'}:
            raise ValueError('Incomplete input-path validation')
        for candidate in CANDIDATES:
            path = case['paths'][candidate]
            if (not np.isfinite(path['tensor_max_abs_error'])
                    or path['tensor_max_abs_error'] > report['limits']['tensor_atol']
                    or path['detections']['expected_object_iou'] < .5
                    or not path['matched_ious']
                    or min(path['matched_ious']) < report['limits']['detection_iou']):
                raise ValueError('Failed numerical or semantic evidence')
    path = root/'samples.csv'
    if not path.exists():
        path = path.with_suffix('.csv.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    values = defaultdict(list)
    expected_order = ((fixture,block,index,candidate)
        for fixture in FIXTURES for block in range(report['blocks'])
        for candidate in CANDIDATES[block%3:]+CANDIDATES[:block%3]
        for index in range(report['samples_per_block']))
    count = 0
    with opener(path,'rt',newline='') as file:
        for row in csv.DictReader(file):
            expected = next(expected_order,None)
            actual = (row['fixture'],int(row['block']),int(row['sample']),row['candidate'])
            if actual != expected:
                raise ValueError('Missing, duplicate, or out-of-order timing samples')
            timing = np.asarray([float(row[field]) for field in FIELDS])
            if not np.isfinite(timing).all() or (timing <= 0).any():
                raise ValueError('Invalid measured timing')
            if abs(timing[:3].sum()-timing[3]) > .01:
                raise ValueError('Component timing differs from complete GPU interval')
            values[(row['fixture'],row['candidate'])].append(timing)
            count += 1
    if next(expected_order,None) is not None:
        raise ValueError('Truncated sample file')
    results = {}
    for fixture in FIXTURES:
        results[fixture] = {}
        for candidate in CANDIDATES:
            array = np.asarray(values[(fixture,candidate)])
            results[fixture][candidate] = {'samples':len(array),'latency':{
                field:dict(zip(('p50','p95','p99'),np.percentile(array[:,i],[50,95,99]).tolist()))
                for i,field in enumerate(FIELDS)},
                'serial_images_per_second':float(1000/array[:,-1].mean())}
        for baseline in ('torch','cvcuda'):
            results[fixture]['cpg_vs_'+baseline] = {
                field:{quantile:100*(1-results[fixture]['cpg']['latency'][field][quantile]/results[fixture][baseline]['latency'][field][quantile])
                       for quantile in ('p50','p95','p99')} for field in FIELDS}
    with opener(path,'rb') as file:
        sample_hash = hashlib.sha256(file.read()).hexdigest()
    return {'status':'verified','samples':count,'source_revision':report['revision'],
            'gpu':report['gpu'],'samples_sha256':sample_hash,'results':results,
            'scope':'serial resident-input comparison; not camera latency, sustained throughput, or a soak'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    summary = summarize(args.root)
    with args.output.open('x') as file:
        json.dump(summary,file,indent=2)
        file.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
