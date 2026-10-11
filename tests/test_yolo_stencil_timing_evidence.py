"""Reject incomplete hardware evidence; synthetic records are not GPU validation."""
import csv
import hashlib
import sys
import importlib.util
import json
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from verify_yolo_stream_ownership import FIXTURES

spec = importlib.util.spec_from_file_location('stencil_summary',Path(__file__).parents[1]/'scripts/summarize_yolo_stencil.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def bundle(tmp_path):
    checks = [dict(fixture=fixture,size=size,path=path,inference='passed',tensor_max_error=0)
              for fixture in ('a','b') for size in (3,5,7,9) for path in ('cpg','pytorch')]
    acceptance_checks = []
    for fixture in FIXTURES:
        for size in (3,5,7,9):
            for variant in ('positive','negative'):
                for layout in ('contiguous','reverse_channels'):
                    for execution in ('sync','submit'):
                        acceptance_checks.append(dict(fixture=fixture,size=size,variant=variant,layout=layout,execution=execution,
                            tensor_exact=True,dense_output=True,detections=True,retained_output_exact=True if acceptance_checks else None,
                            plan=dict(cuda_launches=2,temporary_arrays=1,stencil_implementation='specialized-direct-experimental')))
    acceptance=json.dumps(dict(status='passed',drain_status='passed',dirty=False,mode='stencil-inference-v1',
        revision='a'*40,engine_sha256='b'*64,checks=acceptance_checks)).encode()
    (tmp_path/'acceptance.json').write_bytes(acceptance)
    (tmp_path/'results.json').write_text(json.dumps(dict(status='passed',drain_status='passed',dirty=False,
        mode='stencil-comparison-v1',revision='a'*40,engine_sha256='b'*64,blocks=2,samples=100,
        limits=dict(tensor_atol=.0005),checks=checks,acceptance_sha256=hashlib.sha256(acceptance).hexdigest())))
    with (tmp_path/'timings.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['fixture','size','block','sample','path','preprocess_ms','inference_ms','decode_ms','gpu_total_ms','host_ms'])
        for check in checks:
            for block in range(2):
                for sample in range(100):
                    writer.writerow([check['fixture'],check['size'],block,sample,check['path'],1,2,3,6,7])
    return tmp_path


def test_complete_synthetic_coverage(tmp_path):
    result=module.summarize(bundle(tmp_path))
    assert result['completed_inference_checks']==3200
    assert result['samples'][0]['metrics']['host_ms']['p99']==7


@pytest.mark.parametrize('fault', ['missing','duplicate','nan','failed-inference','wrong-acceptance'])
def test_faulted_evidence_is_rejected(tmp_path,fault):
    root=bundle(tmp_path)
    path=root/'timings.csv'; lines=path.read_text().splitlines()
    if fault=='missing':path.write_text('\n'.join(lines[:-1])+'\n')
    if fault=='duplicate':path.write_text('\n'.join(lines+[lines[-1]])+'\n')
    if fault=='nan':
        row=lines[-1].split(','); row[-1]='nan'; lines[-1]=','.join(row)
        path.write_text('\n'.join(lines)+'\n')
    if fault=='failed-inference':
        report=json.loads((root/'results.json').read_text());report['checks'][0]['inference']='failed'
        (root/'results.json').write_text(json.dumps(report))
    if fault=='wrong-acceptance':
        report=json.loads((root/'results.json').read_text());report['engine_sha256']='c'*64
        (root/'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):module.summarize(root)
