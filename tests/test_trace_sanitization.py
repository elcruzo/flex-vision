import sqlite3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sanitize_trace import sanitize


def test_environment_is_removed_without_changing_gpu_events(tmp_path):
    source=tmp_path/'source.sqlite';target=tmp_path/'sanitized.sqlite'
    secret='rpa_this_is_a_fake_test_credential'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE TARGET_INFO_SYSTEM_ENV (value TEXT)')
        db.execute('INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES (?)',(secret,))
        db.execute('CREATE TABLE StringIds (id INTEGER,value TEXT)')
        db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'KEY='+secret),(2,'preprocess')])
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER)')
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,20)')
    result=sanitize(source,target)
    assert result['redacted_strings']==1
    assert secret.encode() not in target.read_bytes()
    assert secret.encode() in source.read_bytes()
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT * FROM TARGET_INFO_SYSTEM_ENV').fetchall()==[]
        assert db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall()==[(10,20)]
        assert db.execute('SELECT value FROM StringIds WHERE id=2').fetchone()[0]=='preprocess'
