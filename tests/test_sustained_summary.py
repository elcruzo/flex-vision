"""Reject corrupt benchmark evidence without needing a CUDA device."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('sustained_summary', Path(__file__).parents[1]/'scripts/summarize_sustained.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.fixture
def bundle(tmp_path):
    report = {'status': 'passed', 'revision': 'fixture', 'scope': 'synthetic verifier test',
              'runs': [], 'performance_hypothesis': {'passed': False}}
    for cameras in (1, 4):
        for strategy in ('baseline', 'roi'):
            name = f'{cameras}-{strategy}.csv'
            rows = [{'frame': i, 'lane': i, 'variant': 0, 'arrival_s': i/cameras,
                     'submission_s': i/cameras, 'completion_s': i/cameras+.01,
                     'arrival_to_completion_ms': 10, 'gpu_pipeline_ms': 8, 'status': 'completed'}
                    for i in range(cameras)]
            with (tmp_path/name).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report['runs'].append({'cameras': cameras, 'repeat': 0, 'strategy': strategy,
                'fps_per_camera': 1, 'seconds': 1, 'samples': name, 'queue_depth_per_camera': 2,
                'offered': cameras, 'completed': cameras, 'dropped': 0, 'max_inflight': cameras,
                'failed_frames': [], 'elapsed_with_drain_s': 1, 'completed_per_second_with_drain': cameras,
                'arrival_to_completion_ms': dict(p50=10,p95=10,p99=10),
                'allocated_start': 100, 'allocated_after_drain': 100, 'peak_allocated': 200, 'peak_reserved': 300})
    (tmp_path/'report.json').write_text(json.dumps(report))
    return tmp_path


def test_recomputes_per_camera_and_keeps_failed_performance(bundle):
    result = summary.summarize(bundle)
    assert result['status'] == 'verified'
    assert result['performance_passed'] is False
    assert [lane['completed'] for lane in result['runs'][-1]['per_camera']] == [1]*4


def test_duplicate_frame_is_rejected(bundle):
    path = bundle/'4-roi.csv'
    lines = path.read_text().splitlines()
    lines[-1] = lines[-2]
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='duplicate'):
        summary.summarize(bundle)


def test_changed_latency_report_is_rejected(bundle):
    path = bundle/'report.json'
    report = json.loads(path.read_text())
    report['runs'][0]['arrival_to_completion_ms']['p99'] = 5
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='percentiles'):
        summary.summarize(bundle)


def test_allocation_growth_remains_visible_despite_correctness(bundle):
    path = bundle/'report.json'
    report = json.loads(path.read_text())
    report['runs'][0]['allocated_after_drain'] = 101
    path.write_text(json.dumps(report))
    result = summary.summarize(bundle)
    assert result['memory_growth_flags']['within_blocks'] == ['1-baseline-0']


def test_failed_run_cannot_be_summarized_as_passed(bundle):
    path = bundle/'report.json'
    report = json.loads(path.read_text())
    report['status'] = 'failed'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='correctness-passing'):
        summary.summarize(bundle)
