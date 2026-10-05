"""Create one disposable L4 Pod and terminate it from an independent runner.

Run only from the manually dispatched lease workflow. No laptop timer is used.
The receipt binds cleanup to the exact creation response, never a name search.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

IMAGE = 'runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster'
GPU = 'NVIDIA L4'
RECEIPT = Path('lease.json')


def log(event, **fields):
    print(json.dumps({'time':datetime.now(timezone.utc).isoformat(), 'event':event, **fields}), flush=True)


def api(method, path, body=None):
    key = os.environ.get('RUNPOD_API_KEY', '')
    if not key:
        raise RuntimeError('Missing Runpod credential')
    request = Request('https://api.runpod.io/v2/' + path, method=method,
                      headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'},
                      data=None if body is None else json.dumps(body).encode())
    try:
        with urlopen(request, timeout=20) as response:
            data = response.read()
            return json.loads(data) if data else None
    except HTTPError as exc:
        if exc.code == 404 and method in ('GET','DELETE'):
            return None
        # Do not log request headers, credentials, or arbitrary response bodies.
        raise RuntimeError(f'Runpod {method} failed with HTTP {exc.code}') from None


def verify(pod, receipt):
    if pod['id'] != receipt['id'] or pod.get('env',{}).get('CPG_LEASE_RUN') != receipt['run']:
        raise RuntimeError('Pod identity does not match this creation receipt')


def terminate(receipt):
    """Retry transient API failures, and require absence after deletion."""
    for attempt in range(12):
        try:
            pod = api('GET', 'pods/'+receipt['id'])
            if pod is None:
                log('termination_verified', pod=receipt['id'])
                return
            verify(pod, receipt)
            api('DELETE', 'pods/'+receipt['id'])
            if api('GET', 'pods/'+receipt['id']) is None:
                log('termination_verified', pod=receipt['id'])
                return
        except Exception as exc:
            log('cleanup_retry', attempt=attempt+1, error=type(exc).__name__)
        time.sleep(5)
    raise RuntimeError('Pod termination could not be verified; manual cleanup is required')


def create(minutes, run):
    if minutes not in (2,30) or not re.fullmatch(r'[0-9]+-[0-9]+',run):
        raise ValueError('Expected a 2- or 30-minute lease and a GitHub run identifier')
    if RECEIPT.exists():
        raise RuntimeError('Creation receipt already exists; refusing another Pod')
    inventory = api('GET','pods')
    if inventory['pods']:
        raise RuntimeError('Account already has Pods; refusing overlapping rental')
    catalog = api('GET', 'catalog/gpus/'+quote(GPU, safe='')+'?include=AVAILABILITY&product=POD&minCudaVersion=13.0')
    price = catalog['price']['secure']
    available = any(dc['id']=='EU-RO-1' and dc['availability'] in ('LOW','MEDIUM','HIGH') for dc in catalog.get('dataCenters',[]))
    if not available or not 0 < price <= .49:
        raise RuntimeError('L4 capacity or the approved $0.49/hour price is unavailable')
    public_key = os.environ.get('CPG_SSH_PUBLIC_KEY','').strip()
    if not public_key.startswith('ssh-ed25519 '):
        raise ValueError('Missing dedicated SSH public key')
    # Deadline starts before provisioning, not after container startup.
    deadline = time.time()+minutes*60
    log('creating', run=run, deadline=deadline, gpu_hourly_usd=price)
    # Never retry POST: an ambiguous creation response needs manual reconciliation.
    pod = api('POST','pods',{
        'name':'cpg-lease-'+run, 'image':IMAGE, 'cloud':'SECURE',
        'gpu':{'id':GPU,'count':1,'minCudaVersion':'13.0'},
        'dataCenterIds':['EU-RO-1'], 'disk':50, 'ports':['22/tcp'],
        'startSsh':True, 'startJupyter':False,
        'env':{'PUBLIC_KEY':public_key,'CPG_LEASE_RUN':run},
    })
    receipt = {'id':pod['id'],'run':run,'deadline':deadline,'image':IMAGE,'gpu_hourly_usd':price}
    # The only deletion authority is the ID returned by this POST.
    try:
        RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n')
    except Exception:
        terminate(receipt)
        raise
    return receipt


def watch(receipt):
    pod = api('GET','pods/'+receipt['id'])
    if pod is None:
        return
    verify(pod,receipt)
    if float(pod['cost']) > .49:
        raise RuntimeError('Created Pod exceeds the approved GPU rate')
    log('armed', **receipt)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        Path(summary).write_text(f"Pod `{receipt['id']}`. Deadline: {datetime.fromtimestamp(receipt['deadline'],timezone.utc).isoformat()}.\n")
    while time.time() < receipt['deadline']:
        time.sleep(min(10,max(0,receipt['deadline']-time.time())))
        pod = api('GET','pods/'+receipt['id'])
        if pod is None:
            log('manual_termination_verified', pod=receipt['id'])
            return
        verify(pod,receipt)
    log('deadline_reached',pod=receipt['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('lease','cleanup'))
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.mode == 'cleanup':
        if RECEIPT.exists():
            terminate(json.loads(RECEIPT.read_text()))
        return
    receipt = None
    try:
        receipt = create(int(os.environ['LEASE_MINUTES']),os.environ['CPG_RUN'])
        watch(receipt)
    finally:
        if receipt is not None:
            terminate(receipt)


if __name__ == '__main__':
    main()
