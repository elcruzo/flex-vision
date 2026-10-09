"""Reject missing capture controls independently of CUDA availability."""
from pathlib import Path
import sqlite3
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from verify_yolo_trace import verify


def fixture(path):
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER);')
        db.execute('INSERT INTO NVTX_EVENTS VALUES(0,10,?)',('known_4096_byte_d2h_control',))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(1,9,2,4096)')
        for index,name in enumerate(('cpg','torch','cvcuda')):
            for frame in range(2):
                start=20+20*(2*index+frame)
                db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?)',(start,start+10,'yolo_'+name+'_complete'))
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)',(start+1,start+2))
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,2,8)',(start+3,start+4))


def test_control_and_small_copies_are_reported(tmp_path):
    path=tmp_path/'trace.sqlite';fixture(path)
    result=verify(path)
    assert result['ranges']['cpg'][0]['copies'][0]['bytes'] == 8
    assert result['known_control']['captured']


@pytest.mark.parametrize('mutation',[
    'DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes=4096',
    'UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=4095 WHERE bytes=4096',
    "DELETE FROM NVTX_EVENTS WHERE text='yolo_cpg_complete'",
    'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL'])
def test_incomplete_evidence_is_rejected(tmp_path,mutation):
    path=tmp_path/'trace.sqlite';fixture(path)
    with sqlite3.connect(path) as db: db.execute(mutation)
    with pytest.raises(ValueError): verify(path)
