"""Verify a continuous inspection soak from raw frame records."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def percentiles(values):
    return dict(zip(('p50', 'p95', 'p99'), np.percentile(values, [50, 95, 99]).tolist())) if values else None


def summarize(root):
    report = json.loads((root / 'report.json').read_text())
    if report.get('kind') != 'continuous-soak' or len(report['runs']) != 1:
        raise ValueError('Require one continuous soak run')
    run = report['runs'][0]
    if (run['seconds'], run['cameras'], run['fps_per_camera'], run['queue_depth_per_camera']) != (1800, 4, 60, 2):
        raise ValueError('Require the declared 30-minute four-camera workload')
    if run['strategy'] != 'roi' or run['repeat'] != 0 or run['failed_frames'] or run['status'] != 'passed':
        raise ValueError('Require a correct ROI run')
    if not report['conditioning'] or not report['conditioning'][-1]['thermal']['stable']:
        raise ValueError('Missing thermal stabilization')
    path = root / run['samples']
    if path.resolve().parent != root.resolve():
        raise ValueError('Sample path escapes the result directory')
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    minutes = [{'minute': i, 'offered': 0, 'completed': 0, 'drops': Counter(), 'latencies': []} for i in range(30)]
    lanes = [{'lane': i, 'offered': 0, 'completed': 0, 'drops': Counter(), 'latencies': []} for i in range(4)]
    all_latencies, gpu_latencies = [], []
    early_latencies, late_latencies = [], []
    drops = Counter()
    count, completed, within_window = 0, 0, 0
    latest_completion = 0
    pending = [[] for _ in range(4)]
    with opener(path, 'rt', newline='') as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if int(row['frame']) != index or int(row['lane']) != index % 4 or int(row['variant']) != (index // 4) % 2:
                raise ValueError('Missing, duplicate, or incorrect frame identity')
            arrival = float(row['arrival_s'])
            submission = float(row['submission_s'])
            if not math.isfinite(submission) or submission < arrival or not math.isclose(arrival, index / 240, abs_tol=1e-9):
                raise ValueError('Invalid arrival or submission time')
            if index >= 432000:
                raise ValueError('Too many offered frames')
            minute, lane = minutes[index // 14400], lanes[index % 4]
            count += 1
            for group in (minute, lane):
                group['offered'] += 1
            status = row['status']
            if status == 'completed':
                finish = float(row['completion_s'])
                latency = float(row['arrival_to_completion_ms'])
                gpu = float(row['gpu_pipeline_ms'])
                if not all(math.isfinite(x) for x in (finish, latency, gpu)) or finish < submission or gpu <= 0:
                    raise ValueError('Invalid completion timing')
                if not math.isclose(latency, (finish - arrival) * 1000, rel_tol=1e-9, abs_tol=1e-7):
                    raise ValueError('Latency differs from raw timestamps')
                if finish > run['elapsed_with_drain_s'] + 1e-6:
                    raise ValueError('Completion after reported drain')
                latest_completion = max(latest_completion, finish)
                camera = index % 4
                pending[camera] = [end for end in pending[camera] if end > submission]
                pending[camera].append(finish)
                if len(pending[camera]) > 2:
                    raise ValueError('Raw timings exceed the per-camera queue bound')
                completed += 1
                within_window += finish <= 1800
                all_latencies.append(latency)
                if arrival < 300:
                    early_latencies.append(latency)
                if arrival >= 1500:
                    late_latencies.append(latency)
                gpu_latencies.append(gpu)
                for group in (minute, lane):
                    group['completed'] += 1
                    group['latencies'].append(latency)
            elif status in ('dropped_dispatch_late', 'dropped_queue_full'):
                if any(row[field] for field in ('completion_s', 'arrival_to_completion_ms', 'gpu_pipeline_ms')):
                    raise ValueError('Dropped frame has completion timing')
                drops[status] += 1
                for group in (minute, lane):
                    group['drops'][status] += 1
            else:
                raise ValueError('Unknown frame status')
    if count != 432000 or count != run['offered'] or completed != run['completed'] or count-completed != run['dropped']:
        raise ValueError('Offered or completed counts differ')
    if not completed or within_window != run['completed_within_window'] or not 0 < run['max_inflight'] <= 8:
        raise ValueError('Invalid completion count or queue bound')
    if not 1800 <= run['elapsed_with_drain_s'] <= 1830:
        raise ValueError('Run did not span the required continuous duration')
    if any(drops[key] != run['drop_reasons'][key] for key in ('dropped_dispatch_late', 'dropped_queue_full')):
        raise ValueError('Drop counts differ')
    measured = percentiles(all_latencies)
    if any(not math.isclose(measured[k], run['arrival_to_completion_ms'][k], rel_tol=1e-9) for k in measured):
        raise ValueError('Percentiles differ')
    for group in minutes + lanes:
        group['arrival_to_completion_ms'] = percentiles(group.pop('latencies'))
        group['completed_fps'] = group['completed'] / (60 if 'minute' in group else 1800)
    memory = run['memory_samples']
    if any(b['elapsed_s'] <= a['elapsed_s'] for a, b in zip(memory, memory[1:])):
        raise ValueError('Non-monotonic memory sampling')
    early = [x for x in memory if x['elapsed_s'] < 300]
    late = [x for x in memory if 1500 <= x['elapsed_s'] < 1800]
    if not early or not late or memory[0]['elapsed_s'] > 1 or memory[-1]['elapsed_s'] < 1799:
        raise ValueError('Incomplete memory monitoring')
    growth = {'post_drain_allocated': run['allocated_after_drain'] - run['allocated_start'],
              'sampled_allocated_floor': min(x['allocated'] for x in late) - min(x['allocated'] for x in early),
              'sampled_reserved_ceiling': max(x['reserved'] for x in late) - max(x['reserved'] for x in early)}
    memory_passed = all(x <= 1048576 for x in growth.values())
    if growth != report['memory_growth_bytes'] or memory_passed != report['memory_passed']:
        raise ValueError('Memory verdict differs')
    if (report['status'] == 'passed') != memory_passed:
        raise ValueError('Overall verdict differs')
    return {'status': 'verified', 'soak_passed': memory_passed, 'revision': report['revision'],
            'sample_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'summary_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'offered': count, 'completed': completed, 'dropped': count-completed, 'drop_reasons': dict(drops),
            'elapsed_with_drain_s': run['elapsed_with_drain_s'], 'latest_completion_s': latest_completion,
            'completed_fps_with_drain': completed/run['elapsed_with_drain_s'],
            'arrival_to_completion_ms': measured, 'gpu_pipeline_ms': percentiles(gpu_latencies),
            'memory_growth_bytes': growth, 'max_memory_sample_gap_s': max(b['elapsed_s']-a['elapsed_s'] for a,b in zip(memory,memory[1:])),
            'minutes': minutes, 'cameras': lanes,
            'last_to_first_five_minute_p99_ratio': float(np.percentile(late_latencies, 99) / np.percentile(early_latencies, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'soak_passed', 'offered', 'completed', 'dropped')}))


if __name__ == '__main__':
    main()
