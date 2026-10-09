"""Reject incomplete or contradictory stream acceptance evidence."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('stream_evidence', Path(__file__).parents[1] / 'scripts/verify_yolo_stream_ownership.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report():
    return {'status': 'passed', 'dirty': False, 'revision': 'a' * 40, 'engine_sha256': 'b' * 64,
            'stream_handles': [1, 2, 3], 'drain_status': 'passed',
            'cases': [{'name': name, 'input': kind, 'status': 'passed', 'retention': 'passed',
                       'checks': [{'cycle': cycle, 'tensor_exact': True, 'dense_output': True,
                                   'detections': True, 'retained_outputs': True, 'consumer_complete': True}
                                  for cycle in range(4)]}
                      for name in sorted(module.FIXTURES) for kind in ('cupy', 'torch')]}


def test_complete_evidence():
    result = module.verify(report())
    assert result['executions'] == 48
    assert result['cases'] == 12


@pytest.mark.parametrize('fault', ['missing_case', 'duplicate', 'unchecked', 'wrong_cycle', 'shared_stream', 'drain', 'dirty'])
def test_reject_incomplete_evidence(fault):
    data = copy.deepcopy(report())
    if fault == 'missing_case': data['cases'].pop()
    elif fault == 'duplicate': data['cases'].append(data['cases'][0])
    elif fault == 'unchecked': data['cases'][0]['checks'][0]['consumer_complete'] = False
    elif fault == 'wrong_cycle': data['cases'][0]['checks'][0]['cycle'] = 2
    elif fault == 'shared_stream': data['stream_handles'] = [1, 1, 2]
    elif fault == 'drain': data['drain_status'] = 'failed'
    elif fault == 'dirty': data['dirty'] = True
    with pytest.raises(ValueError): module.verify(data)
