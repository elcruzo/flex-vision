"""Fault checks for the experiment's evidence validator, not GPU acceptance."""
import csv
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from summarize_yolo import summarize, CANDIDATES, FIXTURES, FIELDS


def fixture(root):
    entry = {'tensor_max_abs_error':0.,'detections':{'expected_object_iou':.9},'matched_ious':[1.]}
    report = {'status':'passed','dirty':False,'blocks':10,'samples_per_block':1000,'warmup':100,
              'revision':'test-only','gpu':'synthetic evidence checker fixture',
              'limits':{'tensor_atol':.0005,'detection_iou':.95},'cases':[
                  {'id':name,'status':'passed','paths':{key:entry for key in (*CANDIDATES,'torch_dlpack','negative_stride')}}
                  for name in ('astronaut','astronaut-padded','chelsea','chelsea-padded',*FIXTURES)]}
    (root/'results.json').write_text(json.dumps(report))
    with (root/'samples.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['fixture','block','sample','candidate',*FIELDS])
        for name in FIXTURES:
            for block in range(10):
                for candidate in CANDIDATES[block%3:]+CANDIDATES[:block%3]:
                    for index in range(1000):
                        writer.writerow([name,block,index,candidate,1,2,3,6,7])


def test_complete_and_truncated_sample_coverage(tmp_path):
    fixture(tmp_path)
    assert summarize(tmp_path)['samples'] == 60000
    path=tmp_path/'samples.csv'
    lines=path.read_text().splitlines()
    path.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError,match='Truncated'):
        summarize(tmp_path)


@pytest.mark.parametrize('fault', ['nonfinite','duplicate','failed_correctness'])
def test_invalid_evidence_is_rejected(tmp_path,fault):
    fixture(tmp_path)
    path=tmp_path/'samples.csv'
    if fault == 'failed_correctness':
        report=json.loads((tmp_path/'results.json').read_text())
        report['cases'][0]['paths']['cpg']['detections']['expected_object_iou']=.1
        (tmp_path/'results.json').write_text(json.dumps(report))
    else:
        lines=path.read_text().splitlines()
        if fault == 'duplicate':
            lines[2]=lines[1]
        else:
            cells=lines[1].split(',');cells[-1]='nan';lines[1]=','.join(cells)
        path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):
        summarize(tmp_path)


def test_numerical_only_run_is_not_performance_evidence(tmp_path):
    import json
    from summarize_yolo import summarize
    (tmp_path/'results.json').write_text(json.dumps({'run_kind':'validation'}))
    with pytest.raises(ValueError, match='latency comparison'):
        summarize(tmp_path)
