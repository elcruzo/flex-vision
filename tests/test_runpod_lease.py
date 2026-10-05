"""Fault checks for paid-resource cleanup; these do not replace its live test."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location('lease',Path(__file__).parents[1]/'scripts/runpod_lease.py')
lease = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lease)
RECEIPT = {'id':'our-pod','run':'123-1','deadline':0}
POD = {'id':'our-pod','env':{'CPG_LEASE_RUN':'123-1'},'cost':.49}


def test_delete_requires_matching_receipt():
    with patch.object(lease,'api',return_value={'id':'foreign','env':{}}) as api, patch.object(lease.time,'sleep'):
        with pytest.raises(RuntimeError,match='could not be verified'):
            lease.terminate(RECEIPT)
    assert all(call.args[0]=='GET' for call in api.call_args_list)


def test_cleanup_retries_and_checks_absence():
    with patch.object(lease,'api',side_effect=[RuntimeError('outage'),POD,None,None]) as api, patch.object(lease.time,'sleep'):
        lease.terminate(RECEIPT)
    assert [call.args[0] for call in api.call_args_list]==['GET','GET','DELETE','GET']


def test_watch_failure_still_terminates():
    with patch('sys.argv',['lease','lease']), patch.dict('os.environ',LEASE_MINUTES='2',CPG_RUN='123-1'), patch.object(lease,'create',return_value=RECEIPT), patch.object(lease,'watch',side_effect=RuntimeError('outage')), patch.object(lease,'terminate') as terminate:
        with pytest.raises(RuntimeError,match='outage'):
            lease.main()
    terminate.assert_called_once_with(RECEIPT)


def test_invalid_duration_never_calls_provider():
    with patch.object(lease,'api') as api:
        with pytest.raises(ValueError):
            lease.create(31,'123-1')
    api.assert_not_called()


def test_existing_pods_prevent_creation(tmp_path):
    with patch.object(lease,'RECEIPT',tmp_path/'lease.json'), patch.object(lease,'api',return_value={'pods':[POD]}) as api:
        with pytest.raises(RuntimeError,match='already has Pods'):
            lease.create(2,'123-1')
    assert [call.args[0] for call in api.call_args_list]==['GET']
