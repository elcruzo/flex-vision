"""Audit matched serial submission samples without NVIDIA hardware."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import numpy as np

FIELDS = ('dispatch_host_ms', 'preprocess_gpu_ms', 'complete_gpu_ms', 'complete_host_ms')
NAMES = ('astronaut-1080p', 'chelsea-1080p')


def summarize(root):
    root = Path(root)
    report = json.loads((root / 'results.json').read_text())
    if (report.get('status') != 'passed' or report.get('dirty') is not False
            or report.get('drain_status') != 'passed' or report.get('mode') != 'serial-async-v1'
            or (report.get('blocks'), report.get('samples_per_block'), report.get('warmup')) != (10, 1000, 100)
            or report.get('p99_guard_fraction') != .05):
        raise ValueError('Require the complete frozen serial async protocol')
    for key, size in (('revision', 40), ('engine_sha256', 64), ('delayed_report_sha256', 64)):
        if not re.fullmatch('[a-f0-9]{' + str(size) + '}', report.get(key, '')):
            raise ValueError('Missing source, engine, or delayed-gate identity')
    fixtures = report.get('fixtures', [])
    if len(fixtures) != 2 or tuple(f['name'] for f in fixtures) != NAMES:
        raise ValueError('Incorrect fixture coverage')
    if any(f['checked_frames'] != 20000 or f['failed_frames'] != 0 for f in fixtures):
        raise ValueError('Incomplete inference checks')
    path = root / 'samples.csv'
    if not path.exists(): path = Path(str(path) + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as handle:
        rows = list(csv.DictReader(handle))
    identities = [(name, block, index, candidate) for name in NAMES for block in range(10)
                  for candidate in (('sync', 'submit') if block % 2 == 0 else ('submit', 'sync'))
                  for index in range(1000)]
    if len(rows) != len(identities):
        raise ValueError('Incomplete sample coverage')
    for row, expected in zip(rows, identities):
        if (row['fixture'], int(row['block']), int(row['sample']), row['candidate']) != expected or row['correct'] != 'True':
            raise ValueError('Invalid order, identity, or correctness')
        values = [float(row[key]) for key in FIELDS]
        if not np.isfinite(values).all() or min(values) <= 0 or values[0] > values[3] or values[1] > values[2]:
            raise ValueError('Invalid timing boundaries')
    result = {'status': 'samples_verified', 'samples': len(rows), 'fixtures': {},
              'selection': 'within_p99_guard', 'qualification': 'Serial host-wait comparison only; trace, memory, concurrency and repeatability acceptance remain separate'}
    for name in NAMES:
        candidates = {}
        for candidate in ('sync', 'submit'):
            selected = [row for row in rows if row['fixture'] == name and row['candidate'] == candidate]
            def stats(group):
                return {key: dict(zip(('p50', 'p95', 'p99'), np.percentile([float(r[key]) for r in group], [50, 95, 99]).tolist())) for key in FIELDS}
            candidates[candidate] = {'pooled': stats(selected),
                'blocks': [stats([r for r in selected if int(r['block']) == block]) for block in range(10)]}
        regression = candidates['submit']['pooled']['complete_host_ms']['p99'] / candidates['sync']['pooled']['complete_host_ms']['p99'] - 1
        if regression > .05: result['selection'] = 'rejected_p99_regression'
        result['fixtures'][name] = {'timing': candidates, 'complete_host_p99_regression_fraction': regression,
            'complete_host_p50_gain_fraction': 1 - candidates['submit']['pooled']['complete_host_ms']['p50'] / candidates['sync']['pooled']['complete_host_ms']['p50']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as handle:
        json.dump(summarize(args.root), handle, indent=2)


if __name__ == '__main__': main()
