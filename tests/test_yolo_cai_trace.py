"""Reject incomplete CAI capture evidence without claiming GPU execution."""
import json
from pathlib import Path
import sqlite3
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from verify_yolo_cai_trace import verify, PRODUCERS, EXECUTIONS, FIXTURES


def evidence(tmp_path):
    gate = json.loads((Path(__file__).parents[1] / 'docs/experiments/data/031/cai.json').read_text())
    report = dict(status='passed', drain_status='passed', dirty=False, mode='cai-trace-v1',
                  revision=gate['revision'], engine_sha256=gate['engine_sha256'],
                  checks=[dict(fixture=f, producer=p, execution=e, tensor_exact=True, dense_output=True, detections=True)
                          for f in FIXTURES for p in PRODUCERS for e in EXECUTIONS])
    trace = tmp_path / 'trace.sqlite'
    with sqlite3.connect(trace) as db:
        db.executescript('CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT); '
                        'CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER); '
                        'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER); '
                        'CREATE TABLE StringIds(id INTEGER,value TEXT);')
        db.executemany('INSERT INTO StringIds VALUES(?,?)', [(1, 'preprocess'), (2, 'inference')])
        db.execute("INSERT INTO NVTX_EVENTS VALUES(0,10,'known_4096_byte_d2h_control')")
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(1,9,2,4096)')
        for index, row in enumerate(report['checks']):
            start = 20 + index * 20
            name = 'yolo_cai_' + row['producer'] + '_' + row['execution'] + '_complete'
            db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?)', (start, start + 10, name))
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)',
                          [(start + 1, start + 2, 1), (start + 3, start + 4, 2)])
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,2,8)', (start + 5, start + 6))
    return report, gate, trace


def test_small_metadata_copies_remain_visible(tmp_path):
    report, gate, trace = evidence(tmp_path)
    result = verify(report, gate, trace)
    assert result['checked_frames'] == 16
    assert result['trace']['ranges']['cai_explicit_sync'][0]['copies'][0]['bytes'] == 8


@pytest.mark.parametrize('fault', ['missing_case', 'duplicate_case', 'failed_dense', 'wrong_engine',
                                  'wrong_revision', 'missing_preprocess', 'missing_inference', 'payload', 'chunked_payload', 'control'])
def test_incomplete_capture_rejected(tmp_path, fault):
    report, gate, trace = evidence(tmp_path)
    if fault == 'missing_case': report['checks'].pop()
    elif fault == 'duplicate_case': report['checks'][-1] = report['checks'][0]
    elif fault == 'failed_dense': report['checks'][0]['dense_output'] = False
    elif fault == 'wrong_engine': report['engine_sha256'] = 'a' * 64
    elif fault == 'wrong_revision': report['revision'] = 'a' * 40
    else:
        with sqlite3.connect(trace) as db:
            if fault == 'missing_preprocess': db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=1')
            elif fault == 'missing_inference': db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=2')
            elif fault == 'payload': db.execute('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=2457600 WHERE bytes=8')
            elif fault == 'chunked_payload':
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(25,26,1,1024)', [()] * 2400)
            else: db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes=4096')
    with pytest.raises(ValueError): verify(report, gate, trace)
