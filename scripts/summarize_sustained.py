"""Recompute sustained inspection results from every offered frame."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def percentiles(values):
    return dict(zip(('p50', 'p95', 'p99'), np.percentile(values, [50, 95, 99]).tolist())) if values else None


def summarize(root):
    report = json.loads((root/'report.json').read_text())
    if report['status'] != 'passed':
        raise ValueError('Require a complete correctness-passing run')
    result = {'revision': report['revision'], 'runs': [], 'sample_sha256': {},
              'scope': report['scope'], 'status': 'verified'}
    identities = set()
    for run in report['runs']:
        identity = (run['cameras'], run['repeat'], run['strategy'])
        if identity in identities:
            raise ValueError('Duplicate run identity')
        identities.add(identity)
        path = root/run['samples']
        if path.resolve().parent != root.resolve():
            raise ValueError('Sample file must be directly inside the result directory')
        result['sample_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        expected_count = run['cameras'] * run['fps_per_camera'] * run['seconds']
        if len(rows) != expected_count or sorted(int(r['frame']) for r in rows) != list(range(expected_count)):
            raise ValueError('Missing or duplicate offered frames')
        completed = []
        for row in rows:
            index = int(row['frame'])
            if int(row['lane']) != (index+run['repeat']) % run['cameras']:
                raise ValueError('Camera identity differs from the declared schedule')
            if int(row['variant']) != (index//run['cameras']) % 2:
                raise ValueError('Input variant differs from the declared schedule')
            arrival = float(row['arrival_s'])
            if not math.isclose(arrival, index/(run['cameras']*run['fps_per_camera']), abs_tol=1e-9):
                raise ValueError('Arrival differs from the declared schedule')
            submission = float(row['submission_s'])
            if not math.isfinite(submission) or submission < arrival:
                raise ValueError('Invalid submission time')
            if row['status'] == 'completed':
                completion = float(row['completion_s'])
                latency = float(row['arrival_to_completion_ms'])
                gpu = float(row['gpu_pipeline_ms'])
                if (not all(map(math.isfinite, (completion, latency, gpu))) or
                        completion < submission or gpu <= 0 or
                        not math.isclose(latency, (completion-arrival)*1000, abs_tol=1e-6)):
                    raise ValueError('Invalid completed-frame timing')
                if completion > run['elapsed_with_drain_s'] + 1e-6:
                    raise ValueError('Completion exceeds reported duration')
                completed.append(row)
            elif row['status'] not in ('dropped_dispatch_late', 'dropped_queue_full'):
                raise ValueError('Unknown frame status')
            elif any(row[k] for k in ('completion_s','arrival_to_completion_ms','gpu_pipeline_ms')):
                raise ValueError('Dropped frame has completion timings')
        if (run['offered'] != expected_count or run['completed'] != len(completed) or
                run['dropped'] != expected_count-len(completed) or run['failed_frames'] or
                run['max_inflight'] > run['cameras']*run['queue_depth_per_camera']):
            raise ValueError('Frame accounting or correctness report is inconsistent')
        measured = percentiles([float(r['arrival_to_completion_ms']) for r in completed])
        if measured is None:
            raise ValueError('No completed frames')
        for key, value in measured.items():
            if not math.isclose(value, run['arrival_to_completion_ms'][key], abs_tol=1e-6):
                raise ValueError('Latency percentiles differ from raw samples')
        throughput = len(completed)/max(run['seconds'], run['elapsed_with_drain_s'])
        if not math.isclose(throughput, run['completed_per_second_with_drain'], rel_tol=1e-9):
            raise ValueError('Throughput differs from raw samples')
        lanes = []
        for lane in range(run['cameras']):
            offered = [r for r in rows if int(r['lane']) == lane]
            done = [r for r in completed if int(r['lane']) == lane]
            lanes.append({'lane': lane, 'offered': len(offered), 'completed': len(done),
                          'dropped': len(offered)-len(done),
                          'arrival_to_completion_ms': percentiles([float(r['arrival_to_completion_ms']) for r in done])})
        result['runs'].append({**{k: run[k] for k in ('cameras','repeat','strategy','completed','dropped',
            'max_inflight','allocated_start','allocated_after_drain','peak_allocated','peak_reserved')},
            'completed_per_second_with_drain': throughput, 'arrival_to_completion_ms': measured,
            'per_camera': lanes})
    ratios = {1: [], 4: []}
    repeats = sorted({r['repeat'] for r in result['runs']})
    if not repeats or identities != {(c,r,s) for c in (1,4) for r in repeats for s in ('baseline','roi')}:
        raise ValueError('Missing matched workloads')
    for cameras in ratios:
        for repeat in repeats:
            pair = {r['strategy']: r for r in result['runs'] if r['cameras']==cameras and r['repeat']==repeat}
            b, c = pair['baseline'], pair['roi']
            ratios[cameras].append(c['completed_per_second_with_drain']/b['completed_per_second_with_drain']
                if cameras == 4 else c['arrival_to_completion_ms']['p99']/b['arrival_to_completion_ms']['p99'])
    result['median_stress_throughput_ratio'] = float(np.median(ratios[4]))
    result['median_control_p99_ratio'] = float(np.median(ratios[1]))
    result['performance_passed'] = result['median_stress_throughput_ratio'] >= 1.2 and result['median_control_p99_ratio'] <= 1.05
    if result['performance_passed'] != report['performance_hypothesis']['passed']:
        raise ValueError('Performance verdict differs from raw samples')
    result['post_drain_allocated_by_repeat'] = {
        f'{cameras}-{strategy}': [r['allocated_after_drain'] for r in sorted(result['runs'], key=lambda r:r['repeat'])
                                  if r['cameras']==cameras and r['strategy']==strategy]
        for cameras in (1,4) for strategy in ('baseline','roi')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','median_stress_throughput_ratio','median_control_p99_ratio','performance_passed')}))


if __name__ == '__main__':
    main()
