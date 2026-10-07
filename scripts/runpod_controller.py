"""Run a bounded GPU lease without GitHub, using a disposable CPU controller.

Launch from the Mac. The controller executes only these two local scripts.
Do not print full provider responses: they can contain environment variables.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import time

from runpod_lease import api, create, log, terminate, verify, watch

CPU_IMAGE = 'python:3.12-slim-bookworm'
BOOT = '''import os,json,base64,pathlib
root=pathlib.Path('/tmp/cpg-controller'); root.mkdir(exist_ok=True); os.chdir(root)
for name,content in json.loads(base64.b64decode(os.environ['CPG_SOURCE'])).items():
    pathlib.Path(name).write_text(content)
os.execvp('python3',['python3','-u','runpod_controller.py','serve'])
'''


def launch(minutes, receipt_path, controller_data_center='EU-RO-1', gpu_data_center='EU-RO-1'):
    if minutes not in (2,30):
        raise ValueError('Lease must be 2 or 30 minutes')
    if receipt_path.exists():
        raise ValueError('Controller receipt already exists')
    if api('GET','pods')['pods']:
        raise RuntimeError('Account already has Pods; refusing overlapping rental')
    catalog = api('GET','catalog/cpus/cpu3c?include=AVAILABILITY&product=POD&vcpuCount=2')
    rate = catalog['price']['securePerVcpu']*2
    if not 0 < rate <= .06 or not any(d['id']==controller_data_center and d['availability'] in ('LOW','MEDIUM','HIGH') for d in catalog.get('dataCenters',[])):
        raise RuntimeError('Approved CPU rate or requested controller capacity is unavailable')
    public = os.environ.get('CPG_SSH_PUBLIC_KEY','').strip()
    if not public.startswith('ssh-ed25519 '):
        raise ValueError('Missing dedicated SSH public key')
    run = f'{int(time.time())}-{secrets.randbelow(10**12)}'
    root = Path(__file__).resolve().parent
    sources = {name:(root/name).read_text() for name in ('runpod_lease.py','runpod_controller.py')}
    payload = base64.b64encode(json.dumps(sources).encode()).decode()
    secret = api('POST','account/secrets',{
        'name':'cpg-lease-'+run, 'value':os.environ['RUNPOD_API_KEY'],
        'description':'Temporary credential for one CPG CPU controller',
    })
    # Save the exact secret ID before provisioning. Never save the value.
    receipt = {'run':run,'secret_id':secret['id'],'minutes':minutes,'image':CPU_IMAGE,'controller_data_center':controller_data_center,'gpu_data_center':gpu_data_center,
               'source_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}}
    try:
        receipt_path.parent.mkdir(parents=True,exist_ok=True)
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    except Exception:
        api('DELETE','account/secrets/'+secret['id'])
        raise
    try:
        pod = api('POST','pods',{
            'name':'cpg-controller-'+run, 'image':CPU_IMAGE, 'cloud':'SECURE',
            'cpu':{'id':'cpu3c','vcpuCount':2}, 'dataCenterIds':[controller_data_center],
            'disk':2,'ports':[], 'entrypoint':['python3','-u','-c',BOOT], 'cmd':[],
            'env':{'CPG_CONTROL_API_KEY':'{{ RUNPOD_SECRET_'+secret['name']+' }}',
                   'CPG_SECRET_ID':secret['id'],'CPG_LEASE_RUN':run,'CPG_SOURCE':payload,
                   'CPG_SSH_PUBLIC_KEY':public,'LEASE_MINUTES':str(minutes),'CPG_GPU_DATA_CENTER':gpu_data_center,
                   'CPG_DISPATCH_EXPIRES':str(int(time.time()+600))},
        })
    except Exception:
        # POST could have succeeded. Keep the secret/receipt for reconciliation.
        log('creation_uncertain',run=run,secret_id=secret['id'])
        raise
    receipt['id'] = pod['id']
    receipt['cpu_hourly_usd'] = pod['cost']
    try:
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    except Exception:
        terminate({'id':pod['id'],'run':run})
        api('DELETE','account/secrets/'+secret['id'])
        raise
    log('controller_created',**receipt)
    if not 0 < float(pod['cost']) <= .06:
        terminate({'id':pod['id'],'run':run})
        api('DELETE','account/secrets/'+secret['id'])
        raise RuntimeError('Controller exceeds the approved CPU rate')


def serve():
    """Run entirely on the CPU Pod, including creation and deadline cleanup."""
    own = {'id':os.environ['RUNPOD_POD_ID'],'run':os.environ['CPG_LEASE_RUN']}
    gpu = None
    try:
        pod = api('GET','pods/'+own['id'])
        verify(pod,own)
        if pod.get('cpu',{}).get('id') != 'cpu3c' or not 0 < float(pod['cost']) <= .06:
            raise RuntimeError('Controller identity or rate mismatch')
        if not 0 < int(os.environ['CPG_DISPATCH_EXPIRES'])-time.time() <= 600:
            raise RuntimeError('Controller startup expired; refusing GPU creation')
        log('controller_ready',**own)
        gpu = create(int(os.environ['LEASE_MINUTES']),own['run'],controller_id=own['id'],
                     data_center=os.environ.get('CPG_GPU_DATA_CENTER','EU-RO-1'))
        watch(gpu)
    finally:
        try:
            if gpu is not None:
                terminate(gpu)
        finally:
            try:
                api('DELETE','account/secrets/'+os.environ['CPG_SECRET_ID'])
                log('temporary_secret_deleted')
            finally:
                # External readback is required: self-deletion can interrupt this call.
                terminate(own)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('launch','serve'))
    parser.add_argument('--minutes',type=int,choices=(2,30),default=2)
    parser.add_argument('--controller-data-center',default='EU-RO-1')
    parser.add_argument('--gpu-data-center',default='EU-RO-1')
    parser.add_argument('--receipt',type=Path,default=Path('.cache/runpod/controller.json'))
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    if args.mode == 'launch':
        launch(args.minutes,args.receipt,args.controller_data_center,args.gpu_data_center)
    else:
        serve()


if __name__ == '__main__':
    main()
