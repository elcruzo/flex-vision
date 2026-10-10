"""Check unattended export and exact cleanup without renting hardware."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import runpod_experiment as experiment


def test_export_precedes_exact_gpu_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup = tmp_path/'setup.sh'
    setup.write_text('true\n')
    receipt_path = tmp_path/'receipt.json'
    receipt = {'id': 'cpu', 'run': '123-456', 'secret_id': 'secret'}
    pod = {'id': 'gpu', 'env': {'CPG_LEASE_RUN': '123-456','CPG_LEASE_DEADLINE':str(time.time()+3500)},
           'ssh': {'direct': {'username': 'root', 'host': 'host', 'port': 22}}}
    relative = Path('benchmark-results/iteration-test/result.json')
    data = b'{"status":"passed"}\n'

    def launch(*args):
        receipt_path.write_text(json.dumps(receipt))

    def command(args, **kwargs):
        if args[0]=='ssh' and 'nohup' in args[-1]:
            assert 'export CPG_LEASE_DEADLINE=' in args[-1]
        if args[0] == 'scp':
            relative.parent.mkdir(parents=True)
            relative.write_bytes(data)
        return subprocess.CompletedProcess(args, 0, stdout='CPG_EXIT=0\n')

    def cleanup(value):
        assert value == {'id': 'gpu', 'run': '123-456'}
        assert json.loads((relative.parent/'export-verification.json').read_text())['status'] == 'verified'

    with patch.object(experiment, 'launch', side_effect=launch), patch.object(
            experiment, 'api', side_effect=[{'pods': [pod]}, None, None]), patch.object(
            experiment.subprocess, 'run', side_effect=command), patch.object(
            experiment.subprocess, 'check_output', return_value=hashlib.sha256(data).hexdigest()+'  '+str(relative)+'\n'), patch.object(
            experiment, 'terminate', side_effect=cleanup) as terminate:
        experiment.coordinate(receipt_path, setup, 'iteration-test', 'EUR-IS-1')
    terminate.assert_called_once()


def test_ended_controller_does_not_wait_for_absent_gpu(tmp_path, monkeypatch):
    import pytest
    monkeypatch.chdir(tmp_path)
    setup = tmp_path/'setup.sh'
    setup.write_text('true\n')
    receipt_path = tmp_path/'receipt.json'
    def launch(*args):
        receipt_path.write_text(json.dumps({'id':'cpu','secret_id':'secret','run':'123-456'}))
    with patch.object(experiment,'launch',side_effect=launch), patch.object(
            experiment,'api',side_effect=[{'pods':[]},None,None,{'pods':[]},None,None]), patch.object(
            experiment,'terminate') as terminate, patch.object(experiment.time,'sleep') as sleep:
        with pytest.raises(RuntimeError,match='cleanup verified'):
            experiment.coordinate(receipt_path,setup,'iteration-test','EU-CZ-1')
        sleep.assert_not_called()
        terminate.assert_called_once_with({'id':'cpu','run':'123-456'})


def test_startup_timeout_removes_controller_secret_and_raced_gpu(tmp_path,monkeypatch):
    import pytest
    monkeypatch.chdir(tmp_path)
    setup=tmp_path/'setup.sh';setup.write_text('true\n')
    receipt_path=tmp_path/'receipt.json'
    receipt={'id':'cpu','secret_id':'secret','run':'123-456'}
    pending={'id':'late-gpu','env':{'CPG_LEASE_RUN':'123-456'}}
    def launch(*args): receipt_path.write_text(json.dumps(receipt))
    with patch.object(experiment,'launch',side_effect=launch), patch.object(
            experiment.time,'monotonic',side_effect=[0,601]), patch.object(
            experiment,'api',side_effect=[None,{'pods':[pending]},None,None]) as api, patch.object(
            experiment,'terminate') as terminate:
        with pytest.raises(TimeoutError):
            experiment.coordinate(receipt_path,setup,'iteration-test','EU-CZ-1')
    assert [call.args[0]['id'] for call in terminate.call_args_list] == ['cpu','late-gpu']
    assert api.call_args_list[0].args == ('DELETE','account/secrets/secret')


def test_wall_deadline_stops_polling_after_mac_pause(tmp_path, monkeypatch):
    import pytest
    monkeypatch.chdir(tmp_path)
    setup = tmp_path/'setup.sh'; setup.write_text('true\n')
    receipt_path = tmp_path/'receipt.json'
    receipt = {'id':'cpu','run':'123-456','secret_id':'secret'}
    pod = {'id':'gpu','env':{'CPG_LEASE_RUN':'123-456','CPG_LEASE_DEADLINE':'3500'},
           'ssh':{'direct':{'username':'root','host':'host','port':22}}}
    def launch(*args): receipt_path.write_text(json.dumps(receipt))
    with patch.object(experiment,'launch',side_effect=launch), patch.object(
            experiment,'api',side_effect=[{'pods':[pod]},None,None]), patch.object(
            experiment,'verify'), patch.object(experiment,'terminate') as terminate, patch.object(
            experiment.time,'monotonic',return_value=0), patch.object(
            experiment.time,'time',side_effect=[100,100,3600]), patch.object(
            experiment.time,'sleep') as sleep, patch.object(experiment.subprocess,'run') as run:
        with pytest.raises(TimeoutError,match='cloud deadline'):
            experiment.coordinate(receipt_path,setup,'iteration-test','EU-RO-1')
    assert run.call_count == 2  # Upload and submit only. No post-expiration SSH poll.
    sleep.assert_not_called()
    terminate.assert_called_once_with({'id':'gpu','run':'123-456'})
