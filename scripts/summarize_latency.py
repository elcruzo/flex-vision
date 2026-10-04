"""Recompute a matched latency comparison from an uninstrumented sample bundle."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS = ('preprocess_gpu_ms', 'network_gpu_ms', 'decode_gpu_ms', 'total_gpu_ms', 'total_host_ms')
CANDIDATES = ('torch_unfused', 'cpg')


def summarize(directory):
    report = json.loads((directory/'report.json').read_text())
    if report['status'] != 'passed' or report['instrumented']:
        raise ValueError('A passed, uninstrumented run is required')
    data = (directory/'samples.csv').read_bytes()
    if hashlib.sha256(data).hexdigest() != report['samples_sha256']:
        raise ValueError('Sample checksum mismatch')
    with (directory/'samples.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    repeats, count = report['repeats'], report['samples_per_repeat']
    if repeats < 1 or count < 1 or len(rows) != repeats*count*len(CANDIDATES):
        raise ValueError('Unexpected sample count')
    groups = {(repeat, candidate): [] for repeat in range(repeats) for candidate in CANDIDATES}
    seen = set()
    for row in rows:
        repeat, sample, candidate = int(row['repeat']), int(row['sample']), row['candidate']
        key = (repeat, candidate, sample)
        if (repeat,candidate) not in groups or not 0 <= sample < count or key in seen:
            raise ValueError('Invalid or duplicate sample identity')
        seen.add(key)
        values = [float(row[metric]) for metric in METRICS]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('All latency samples must be finite and positive')
        groups[repeat,candidate].append(values)
    if any(len(values) != count for values in groups.values()):
        raise ValueError('Unbalanced candidate samples')

    def percentiles(values):
        # Rows are samples, columns are metrics, percentile output is metric-major.
        return np.percentile(values, [50,95,99], axis=0, method='linear').T

    pooled = {candidate: percentiles([row for repeat in range(repeats) for row in groups[repeat,candidate]])
              for candidate in CANDIDATES}
    paired = np.asarray([100*(1-percentiles(groups[repeat,'cpg'])/percentiles(groups[repeat,'torch_unfused']))
                         for repeat in range(repeats)])
    output = {'scope':report['scope'], 'samples_per_candidate':repeats*count, 'repeats':repeats,
              'percentile_method':'numpy linear', 'metrics':{},
              'interpretation':'Positive reduction favors CPG. Repeat ranges describe observed variation, not confidence intervals.',
              'samples_sha256':report['samples_sha256']}
    for i,metric in enumerate(METRICS):
        output['metrics'][metric] = {
            candidate:dict(zip(('p50','p95','p99'),pooled[candidate][i].tolist())) for candidate in CANDIDATES}
        output['metrics'][metric]['paired_reduction_percent'] = {
            label:{'median':float(np.median(paired[:,i,j])), 'min':float(np.min(paired[:,i,j])), 'max':float(np.max(paired[:,i,j]))}
            for j,label in enumerate(('p50','p95','p99'))}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.output.open('x') as dest:
        json.dump(result,dest,indent=2)
        dest.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
