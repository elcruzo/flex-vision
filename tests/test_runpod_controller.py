"""Check controller placement without creating paid resources."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import runpod_controller as controller


def test_requested_region_and_receipt(tmp_path):
    catalog = {'price':{'securePerVcpu':.03},'dataCenters':[{'id':'EU-NL-1','availability':'HIGH'}]}
    replies = [{'pods':[]},catalog,{'id':'secret','name':'secret-name'},{'id':'cpu','cost':.06}]
    with patch.object(controller,'api',side_effect=replies) as api, patch.dict('os.environ',RUNPOD_API_KEY='test-only',CPG_SSH_PUBLIC_KEY='ssh-ed25519 test'):
        controller.launch(30,tmp_path/'receipt.json','EU-NL-1')
    assert api.call_args_list[-1].args[2]['dataCenterIds'] == ['EU-NL-1']
    import json
    assert json.loads((tmp_path/'receipt.json').read_text())['controller_data_center']=='EU-NL-1'


def test_unavailable_region_never_creates_secret(tmp_path):
    catalog = {'price':{'securePerVcpu':.03},'dataCenters':[]}
    with patch.object(controller,'api',side_effect=[{'pods':[]},catalog]) as api:
        with pytest.raises(RuntimeError,match='capacity'):
            controller.launch(30,tmp_path/'receipt.json','EU-NL-1')
    assert all(call.args[0]=='GET' for call in api.call_args_list)


def test_failed_gpu_is_terminated_before_diagnostic_window():
    events = []
    own = {'id':'cpu','env':{'CPG_LEASE_RUN':'123-456'},'cpu':{'id':'cpu3c'},'cost':.06}
    gpu = {'id':'gpu','run':'123-456'}
    def stop(receipt):
        events.append(('stop',receipt['id']))
    with patch.dict('os.environ',RUNPOD_POD_ID='cpu',CPG_LEASE_RUN='123-456',
                    CPG_DISPATCH_EXPIRES='1100',LEASE_MINUTES='60',CPG_SECRET_ID='secret'), patch.object(
            controller,'api',side_effect=[own,None]), patch.object(controller.time,'time',return_value=1000), patch.object(
            controller,'create',return_value=gpu), patch.object(controller,'watch',side_effect=RuntimeError('test failure')), patch.object(
            controller,'terminate',side_effect=stop), patch.object(controller.time,'sleep',side_effect=lambda seconds:events.append(('wait',seconds))):
        with pytest.raises(RuntimeError,match='test failure'):
            controller.serve()
    assert events == [('stop','gpu'),('wait',30),('stop','cpu')]
