"""Continuous inspection soak with thermal conditioning and retained raw evidence."""
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def save_run(output, name, result, rows):
    filename = name + '.csv'
    with (output / filename).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['frame', 'lane', 'variant', 'arrival_s',
            'submission_s', 'status', 'completion_s', 'arrival_to_completion_ms', 'gpu_pipeline_ms'])
        writer.writeheader()
        writer.writerows(rows)
    result['samples'] = filename
    (output / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')


def thermal_window(path):
    rows = list(csv.reader(path.read_text().splitlines()))[-120:]
    if len(rows) != 120:
        raise RuntimeError('Require 120 one-second thermal samples before the soak')
    temperatures = [float(row[1]) for row in rows]
    spread = max(temperatures) - min(temperatures)
    change = float(np.median(temperatures[60:]) - np.median(temperatures[:60]))
    return {'samples': 120, 'range_c': spread, 'median_change_c': change,
            'stable': spread <= 2 and abs(change) <= 1}


def run_soak(run, model, sources, expected, expected_logits, roi, streams, output):
    """Do not split the measured 30 minutes or clear allocator caches."""
    telemetry_path = output / 'telemetry.csv'
    result = {'kind': 'continuous-soak', 'status': 'failed', 'conditioning': [], 'runs': [],
              'policy': {'seconds': 1800, 'cameras': 4, 'fps_per_camera': 60, 'depth': 2,
                         'thermal_range_c': 2, 'thermal_median_change_c': 1,
                         'allocation_growth_allowance_bytes': 1048576}}
    with telemetry_path.open('w') as telemetry:
        process = subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used',
            '--format=csv,noheader,nounits', '-l', '1'], stdout=telemetry)
        try:
            for attempt in range(2):
                block, rows = run(model, sources, expected, expected_logits, roi,
                                  'roi', 4, 60, 300, 2, 0, streams)
                save_run(output, f'conditioning-{attempt}', block, rows)
                del rows
                thermal = thermal_window(telemetry_path)
                result['conditioning'].append({'run': block, 'thermal': thermal})
                print(json.dumps({'event': 'thermal_conditioning', **thermal}), flush=True)
                if block['status'] != 'passed':
                    raise AssertionError('Conditioning failed inference correctness')
                if process.poll() is not None:
                    raise RuntimeError('GPU telemetry stopped unexpectedly')
                if thermal['stable']:
                    break
            else:
                raise RuntimeError('Temperature did not stabilize within ten minutes')
            print(json.dumps({'event': 'continuous_soak_started', 'seconds': 1800,
                              'unix_s': time.time()}), flush=True)
            block, rows = run(model, sources, expected, expected_logits, roi,
                              'roi', 4, 60, 1800, 2, 0, streams)
            save_run(output, 'soak', block, rows)
            result['runs'].append(block)
            early = [x for x in block['memory_samples'] if x['elapsed_s'] < 300]
            late = [x for x in block['memory_samples'] if 1500 <= x['elapsed_s'] < 1800]
            if not early or not late:
                raise AssertionError('Missing memory windows')
            growth = {
                'post_drain_allocated': block['allocated_after_drain'] - block['allocated_start'],
                'sampled_allocated_floor': min(x['allocated'] for x in late) - min(x['allocated'] for x in early),
                'sampled_reserved_ceiling': max(x['reserved'] for x in late) - max(x['reserved'] for x in early)}
            result['memory_growth_bytes'] = growth
            result['memory_passed'] = all(value <= 1048576 for value in growth.values())
            result['status'] = 'passed' if block['status'] == 'passed' and result['memory_passed'] else 'failed'
            # Timing and drop distributions are reported, not a new speed comparison.
            if process.poll() is not None:
                raise RuntimeError('GPU telemetry stopped unexpectedly')
            return result
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (output / 'soak-status.json').write_text(json.dumps(result, indent=2) + '\n')
