"""Start, export, and end one prepared experiment under an independent controller.

This local coordinator must stay awake for artifact export. The cloud controller
still enforces cleanup if this process, its SSH connection, or the Mac fails.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from runpod_controller import launch
from runpod_lease import api, log, terminate, verify


def coordinate(receipt_path, setup, result_name, gpu_region, controller_region='EU-RO-1', gpu='NVIDIA L4'):
    if not re.fullmatch(r'iteration-[a-zA-Z0-9-]+', result_name):
        raise ValueError('Invalid result directory name')
    target = Path('benchmark-results') / result_name
    if target.exists():
        raise ValueError('Result directory already exists')
    setup_bytes = setup.read_bytes()
    launch(60, receipt_path, controller_region, gpu_region, gpu)
    receipt = json.loads(receipt_path.read_text())
    gpu = None
    started = time.monotonic()
    try:
        while time.monotonic() - started < 600:
            pods = api('GET', 'pods')['pods']
            matches = [p for p in pods if p.get('env', {}).get('CPG_LEASE_RUN') == receipt['run']
                       and p['id'] != receipt['id']]
            if len(matches) > 1:
                raise RuntimeError('More than one GPU matches this controller run')
            if matches:
                pod = matches[0]
                gpu = {'id': pod['id'], 'run': receipt['run']}
                verify(pod, gpu)
                direct = pod.get('ssh', {}).get('direct')
                if direct:
                    break
            time.sleep(10)
        else:
            raise TimeoutError('SSH did not become available within ten minutes')
        key = str(Path.home() / '.ssh/cpg_runpod')
        destination = direct['username'] + '@' + direct['host']
        options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                   '-o', 'StrictHostKeyChecking=accept-new', '-i', key]
        ssh = ['ssh', *options, '-p', str(direct['port']), destination]
        scp = ['scp', '-q', *options, '-P', str(direct['port'])]
        # Do not retry experiment submission after an ambiguous SSH result.
        subprocess.run(ssh + ['cat > /workspace/cpg-experiment.sh'], input=setup_bytes,
                       check=True, timeout=30)
        command = 'bash /workspace/cpg-experiment.sh; code=$?; echo "$code" > /workspace/cpg-experiment.exit'
        subprocess.run(ssh + ['nohup bash -c ' + shlex.quote(command) +
            ' > /workspace/cpg-experiment.log 2>&1 < /dev/null &'], check=True, timeout=30)
        log('experiment_submitted', pod=gpu['id'], result=result_name,
            setup_sha256=hashlib.sha256(setup_bytes).hexdigest())
        while time.monotonic() - started < 3300:
            try:
                completed = subprocess.run(ssh + [
                    'tail -2 /workspace/cpg-experiment.log; if test -f /workspace/cpg-experiment.exit; then '
                    'printf "CPG_EXIT="; cat /workspace/cpg-experiment.exit; fi'],
                    capture_output=True, text=True, timeout=30, check=True)
                print(completed.stdout, end='', flush=True)
                if 'CPG_EXIT=' in completed.stdout:
                    exit_code = int(completed.stdout.rsplit('CPG_EXIT=', 1)[1].strip())
                    break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                log('ssh_poll_retry', pod=gpu['id'])
            time.sleep(30)
        else:
            raise TimeoutError('Experiment exceeded local export allowance')
        remote = '/workspace/flex-vision/benchmark-results/' + result_name
        # Preserve failed experiments too. Only complete files are hashed.
        subprocess.run(ssh + [f'cp /workspace/cpg-experiment.log {remote}/execution.log'],
                       check=True, timeout=30)
        manifest = subprocess.check_output(ssh + [
            f'cd /workspace/flex-vision; find benchmark-results/{result_name} -type f -print0 | sort -z | xargs -0 sha256sum'],
            timeout=30, text=True)
        subprocess.run(scp + ['-r', destination + ':' + remote, 'benchmark-results/'], check=True, timeout=180)
        checked = []
        for line in manifest.splitlines():
            expected, name = line.split(maxsplit=1)
            path = Path(name)
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError('Manifest path escapes this result directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Export hash mismatch: ' + name)
            checked.append(name)
        (target / 'remote-sha256.txt').write_text(manifest)
        (target / 'export-verification.json').write_text(json.dumps({
            'status': 'verified', 'files': checked, 'experiment_exit_code': exit_code,
            'pod': gpu['id'], 'run': gpu['run']}, indent=2) + '\n')
        log('export_verified', files=len(checked), exit_code=exit_code, pod=gpu['id'])
    finally:
        if gpu is not None:
            terminate(gpu)
        # The controller removes its secret and itself after the GPU disappears.
        # If GPU creation was never observed, keep its independent startup guard.
    for _ in range(12):
        if api('GET', 'pods/' + receipt['id']) is None and api('GET', 'account/secrets/' + receipt['secret_id']) is None:
            log('controller_cleanup_verified', pod=receipt['id'], secret_id=receipt['secret_id'])
            return
        time.sleep(10)
    raise RuntimeError('Controller cleanup requires an external follow-up read')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--result-name', required=True)
    parser.add_argument('--gpu-region', default='EUR-IS-1')
    parser.add_argument('--gpu', choices=('NVIDIA L4','NVIDIA GeForce RTX 4090'), default='NVIDIA L4')
    parser.add_argument('--controller-region', default='EU-RO-1')
    args = parser.parse_args()
    coordinate(args.receipt, args.setup, args.result_name, args.gpu_region, args.controller_region, args.gpu)


if __name__ == '__main__':
    main()
