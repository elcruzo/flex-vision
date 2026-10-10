"""Check selected trace preservation and reject unsanitized inputs."""
import sqlite3
import pytest
from test_yolo_cai_trace import evidence
from verify_yolo_cai_trace import verify, PRODUCERS, EXECUTIONS
from project_yolo_trace import project


def test_projection_preserves_cai_audit_and_excludes_unselected_work(tmp_path):
    report, gate, trace = evidence(tmp_path)
    with sqlite3.connect(trace) as db:
        db.execute("INSERT INTO NVTX_EVENTS VALUES(1000,1100,'unselected')")
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1001,1002,2)')
    target = tmp_path / 'ranges.sqlite'
    candidates = tuple('cai_' + p + '_' + e for p in PRODUCERS for e in EXECUTIONS)
    result = project(trace, target, candidates)
    assert result['rows']['NVTX_EVENTS'] == 17
    assert result['rows']['CUPTI_ACTIVITY_KIND_KERNEL'] == 32
    assert verify(report, gate, target)['checked_frames'] == 16


@pytest.mark.parametrize('fault', ['environment', 'credential'])
def test_unsanitized_projection_rejected(tmp_path, fault):
    _, _, trace = evidence(tmp_path)
    with sqlite3.connect(trace) as db:
        if fault == 'environment':
            db.execute('CREATE TABLE TARGET_INFO_SYSTEM_ENV(value TEXT)')
            db.execute("INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES('private environment')")
        else: db.execute("INSERT INTO StringIds VALUES(3,'rpa_fake_test_credential')")
    with pytest.raises(ValueError, match='Sanitize'):
        project(trace, tmp_path / 'ranges.sqlite', ('cai_explicit_sync',))
