"""Fault injection for capture and allocator evidence, without a GPU claim."""
import csv
import json
from pathlib import Path
import sqlite3
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from verify_yolo_async_diagnostics import verify
from test_yolo_trace import fixture


def evidence(tmp_path):
    trace = tmp_path / 'trace.sqlite'
    fixture(trace)
    with sqlite3.connect(trace) as db:
        db.execute("UPDATE NVTX_EVENTS SET text='yolo_sync_complete' WHERE text='yolo_cpg_complete'")
        db.execute("UPDATE NVTX_EVENTS SET text='yolo_submit_complete' WHERE text='yolo_torch_complete'")
    gate = json.loads((Path(__file__).parents[1] / 'docs/experiments/data/027/delayed-consumer.json').read_text())
    names = ('astronaut-1080p', 'chelsea-1080p')
    report = dict(status='passed', drain_status='passed', mode='serial-async-diagnostics-v1',
                  revision=gate['revision'], engine_sha256=gate['engine_sha256'], dirty=False,
                  blocks=10, samples_per_block=100, warmup=100,
                  fixtures=[dict(name=n, checked_frames=2000, failed_frames=0) for n in names],
                  allocator_checkpoints=[dict(fixture=n, block=b, cupy_used_bytes=0, cupy_total_bytes=100,
                      torch_allocated_bytes=0, torch_reserved_bytes=100) for n in names for b in range(-1, 10)])
    (tmp_path / 'results.json').write_text(json.dumps(report))
    with (tmp_path / 'samples.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('fixture', 'block', 'sample', 'candidate', 'correct'))
        writer.writerows((n,b,i,c,True) for n in names for b in range(10)
                         for c in (('sync','submit') if b % 2 == 0 else ('submit','sync')) for i in range(100))
    return trace, gate, report


def test_growth_is_not_stability(tmp_path):
    trace, gate, report = evidence(tmp_path)
    assert verify(tmp_path, trace, gate)['allocator_selection'] == 'stable'
    report['allocator_checkpoints'][-1]['torch_reserved_bytes'] += 1
    (tmp_path / 'results.json').write_text(json.dumps(report))
    assert verify(tmp_path, trace, gate)['allocator_selection'] == 'growth_observed'


@pytest.mark.parametrize('fault', ['payload', 'missing_frame', 'missing_checkpoint', 'wrong_protocol', 'wrong_engine'])
def test_incomplete_diagnostics_fail(tmp_path, fault):
    trace, gate, report = evidence(tmp_path)
    if fault == 'payload':
        with sqlite3.connect(trace) as db:
            db.execute('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=2457600 WHERE bytes=8')
    elif fault == 'missing_frame':
        path = tmp_path / 'samples.csv'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    elif fault == 'missing_checkpoint': report['allocator_checkpoints'].pop()
    elif fault == 'wrong_protocol': report['mode'] = 'serial-async-v1'
    elif fault == 'wrong_engine': report['engine_sha256'] = '0' * 64
    (tmp_path / 'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError): verify(tmp_path, trace, gate)
