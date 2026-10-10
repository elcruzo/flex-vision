"""Reject mislabeled, incomplete, and regressed async comparison evidence."""
import csv
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from summarize_yolo_async import summarize, FIELDS, NAMES


@pytest.fixture
def evidence(tmp_path):
    report = {'status': 'passed', 'dirty': False, 'drain_status': 'passed', 'mode': 'serial-async-v1',
              'revision': 'a'*40, 'engine_sha256': 'b'*64, 'delayed_report_sha256': 'c'*64,
              'blocks': 10, 'samples_per_block': 1000, 'warmup': 100, 'p99_guard_fraction': .05,
              'fixtures': [{'name': n, 'checked_frames': 20000, 'failed_frames': 0} for n in NAMES]}
    rows = [dict(fixture=n, block=b, sample=i, candidate=c, correct=True,
                 **dict(zip(FIELDS, (.1, .1, .8, 1.)))) for n in NAMES for b in range(10)
            for c in (('sync','submit') if b%2 == 0 else ('submit','sync')) for i in range(1000)]
    def save():
        (tmp_path/'results.json').write_text(json.dumps(report))
        with (tmp_path/'samples.csv').open('w', newline='') as file:
            writer=csv.DictWriter(file,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    save()
    return tmp_path, report, rows, save


def test_complete_comparison_and_tail_rejection(evidence):
    root, report, rows, save = evidence
    assert summarize(root)['samples'] == 40000
    for row in rows:
        if row['candidate'] == 'submit': row['complete_host_ms'] = 1.06
    save()
    assert summarize(root)['selection'] == 'rejected_p99_regression'


@pytest.mark.parametrize('fault', ['mode', 'missing', 'incorrect', 'nonfinite', 'identity', 'drain', 'gate'])
def test_invalid_evidence_rejected(evidence, fault):
    root, report, rows, save = evidence
    if fault == 'mode': report['mode'] = 'sync'
    if fault == 'missing': rows.pop()
    if fault == 'incorrect': rows[0]['correct'] = False
    if fault == 'nonfinite': rows[0]['complete_host_ms'] = float('nan')
    if fault == 'identity': rows[1]['sample'] = 0
    if fault == 'drain': report['drain_status'] = 'failed'
    if fault == 'gate': report.pop('delayed_report_sha256')
    save()
    with pytest.raises(ValueError): summarize(root)


def test_matching_hardware_gate_required():
    from yolo_async_benchmark import require_gate
    from verify_yolo_delayed_consumer import FIXTURES
    report = {'status':'passed','dirty':False,'drain_status':'passed','revision':'a'*40,
              'engine_sha256':'b'*64,'delay_cycles':500000000,
              'cases':[{'name':n,'status':'passed','checks':[dict(input=k,cycle=i,
                  **{key:True for key in ('pending_after_close','rebind_rejected','dense_output','detections','input_tensor_exact','subsequent_tensor_exact')})
                  for k in ('cupy','torch') for i in range(4)]} for n in FIXTURES]}
    assert require_gate(report, 'a'*40, 'b'*64, False)['executions'] == 48
    for revision, engine, dirty in [('c'*40,'b'*64,False),('a'*40,'c'*64,False),('a'*40,'b'*64,True)]:
        with pytest.raises(ValueError, match='matching'):
            require_gate(report, revision, engine, dirty)
