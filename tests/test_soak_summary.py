"""Reject incomplete and altered soak evidence; synthetic rows are not GPU evidence."""
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from summarize_soak import summarize


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp('soak')
    fields = ['frame', 'lane', 'variant', 'arrival_s', 'submission_s', 'status',
              'completion_s', 'arrival_to_completion_ms', 'gpu_pipeline_ms']
    with gzip.open(root/'soak.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(432000):
            arrival = index / 240
            writer.writerow(dict(zip(fields, [index, index % 4, (index // 4) % 2,
                arrival, arrival, 'completed', arrival + .01, 10, 5])))
    report = {'kind': 'continuous-soak', 'revision': 'synthetic-test-only', 'status': 'passed',
        'conditioning': [{'thermal': {'stable': True}}], 'memory_passed': True,
        'memory_growth_bytes': {'post_drain_allocated': 0, 'sampled_allocated_floor': 0, 'sampled_reserved_ceiling': 0},
        'runs': [{'seconds': 1800, 'cameras': 4, 'fps_per_camera': 60, 'queue_depth_per_camera': 2,
                  'started_unix_s': 10000,
                  'strategy': 'roi', 'repeat': 0, 'failed_frames': [], 'status': 'passed', 'samples': 'soak.csv',
                  'offered': 432000, 'completed': 432000, 'dropped': 0, 'completed_within_window': 431998,
                  'elapsed_with_drain_s': 1800.02, 'max_inflight': 4,
                  'drop_reasons': {'dropped_dispatch_late': 0, 'dropped_queue_full': 0},
                  'arrival_to_completion_ms': {'p50': 10, 'p95': 10, 'p99': 10},
                  'allocated_start': 100, 'allocated_after_drain': 100,
                  'memory_samples': [{'elapsed_s': t, 'allocated': 100, 'reserved': 200}
                                     for t in (.01, 299, 1500, 1799.9)]}]}
    (root/'report.json').write_text(json.dumps(report))
    (root/'telemetry.csv').write_text(''.join(
        datetime.fromtimestamp(10000 + i, timezone.utc).strftime('%Y/%m/%d %H:%M:%S.000') + ', 60, 1500, 70, 100, 1100\n'
        for i in range(1801)))
    return root


def test_recomputes_compressed_frames(evidence):
    result = summarize(evidence)
    assert result['soak_passed'] and result['completed'] == 432000
    assert len(result['minutes']) == 30
    assert all(lane['completed'] == 108000 for lane in result['cameras'])
    assert result['telemetry']['samples'] == 1801


@pytest.mark.parametrize('mutation, error', [
    ('duration', '30-minute'), ('count', 'counts differ'), ('latency', 'Percentiles'), ('memory', 'Memory verdict')])
def test_rejects_altered_report(evidence, tmp_path, mutation, error):
    report = json.loads((evidence/'report.json').read_text())
    if mutation == 'duration':
        report['runs'][0]['seconds'] = 1799
    elif mutation == 'count':
        report['runs'][0]['completed'] -= 1
    elif mutation == 'latency':
        report['runs'][0]['arrival_to_completion_ms']['p99'] = 9
    else:
        report['memory_growth_bytes']['post_drain_allocated'] = 123
    # Keep the sample path inside this fixture directory without copying the large file.
    import os
    os.link(evidence/'soak.csv.gz', tmp_path/'soak.csv.gz')
    (tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match=error):
        summarize(tmp_path)
