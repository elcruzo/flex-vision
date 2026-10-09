"""Matched reuse audit rejects missing checks and ineffective instrumentation."""
import csv
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_yolo_reuse import summarize,FIELDS


def test_reuse_evidence_positive_control_and_faults(tmp_path):
    names=('astronaut-1080p','chelsea-1080p')
    fresh={'frames':100,'pool_requests':100,'requested_bytes':245760000,'device_allocations':0,'device_bytes':0}
    reuse={'frames':100,'pool_requests':0,'requested_bytes':0,'device_allocations':0,'device_bytes':0}
    report={'status':'passed','dirty':False,'blocks':10,'samples_per_block':1000,'warmup':100,'scope':'test fixture',
            'fixtures':[{'name':n,'checked_frames':20000,'failed_frames':0,'allocations':{'fresh':dict(fresh),'reuse':dict(reuse)}} for n in names]}
    def save():(tmp_path/'results.json').write_text(json.dumps(report))
    save();rows=[]
    for name in names:
        for block in range(10):
            for candidate in (('fresh','reuse') if block%2==0 else ('reuse','fresh')):
                for sample in range(1000):
                    rows.append(dict(fixture=name,block=block,sample=sample,candidate=candidate,correct=True,
                                     **dict(zip(FIELDS,(.1,.1,1.,1.)))))
    def save_rows():
        with (tmp_path/'samples.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    save_rows();result=summarize(tmp_path)
    assert result['samples']==40000 and result['adoption']=='acceptable'
    report['fixtures'][0]['allocations']['fresh']['pool_requests']=0;save()
    with pytest.raises(ValueError,match='positive control'):summarize(tmp_path)
    report['fixtures'][0]['allocations']['fresh']['pool_requests']=100;save()
    rows[0]['correct']=False;save_rows()
    with pytest.raises(ValueError,match='correctness'):summarize(tmp_path)
    rows[0]['correct']=True;rows[0]['preprocess_host_ms']=float('nan');save_rows()
    with pytest.raises(ValueError,match='timing'):summarize(tmp_path)
    rows[0]['preprocess_host_ms']=.1
    for row in rows:
        if row['candidate']=='reuse':row['complete_host_ms']=1.06
    save_rows()
    assert summarize(tmp_path)['adoption']=='rejected_p99_regression'
