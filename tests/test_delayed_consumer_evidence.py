"""Fault checks for the next hardware evidence audit."""
import importlib.util
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('delayed_audit',Path(__file__).parents[1]/'scripts/verify_yolo_delayed_consumer.py')
module=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
try: spec.loader.exec_module(module)
finally: sys.path.pop(0)


def report():
    return {'status':'passed','dirty':False,'drain_status':'passed','revision':'a'*40,'engine_sha256':'b'*64,'delay_cycles':500000000,
            'cases':[{'name':name,'status':'passed','checks':[{'input':kind,'cycle':cycle,'pending_after_close':True,
                      'rebind_rejected':True,'dense_output':True,'detections':True,'input_tensor_exact':True,'subsequent_tensor_exact':True}
                      for kind in ('cupy','torch') for cycle in range(4)]} for name in sorted(module.FIXTURES)]}


def test_complete_delayed_evidence():
    assert module.verify(report())['executions']==48


@pytest.mark.parametrize('fault',['completed_consumer','missing_cycle','duplicate_fixture','drain_failed'])
def test_reject_wrong_boundary_or_coverage(fault):
    data=report()
    if fault=='completed_consumer': data['cases'][0]['checks'][0]['pending_after_close']=False
    elif fault=='missing_cycle': data['cases'][0]['checks'].pop()
    elif fault=='duplicate_fixture': data['cases'].append(data['cases'][0])
    else: data['drain_status']='failed'
    with pytest.raises(ValueError): module.verify(data)
