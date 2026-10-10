"""Require complete specialized-stencil real-inference evidence."""
import argparse
import json
from pathlib import Path
import re
from verify_yolo_stream_ownership import FIXTURES


def verify(report):
    if report.get('status')!='passed' or report.get('drain_status')!='passed' or report.get('dirty') is not False or report.get('mode')!='stencil-inference-v1':
        raise ValueError('Require complete clean stencil inference and drainage')
    for field,length in [('revision',40),('engine_sha256',64)]:
        if not re.fullmatch('[a-f0-9]{'+str(length)+'}',report.get(field,'')):raise ValueError('Missing source or engine identity')
    expected={(f,s,v,l,e) for f in FIXTURES for s in (3,5,7,9) for v in ('positive','negative')
              for l in ('contiguous','reverse_channels') for e in ('sync','submit')}
    seen=set()
    for i,row in enumerate(report.get('checks',[])):
        identity=tuple(row.get(k) for k in ('fixture','size','variant','layout','execution'))
        if identity not in expected or identity in seen:raise ValueError('Unexpected or duplicate stencil scenario')
        seen.add(identity)
        plan=row.get('plan',{})
        if (any(row.get(k) is not True for k in ('tensor_exact','dense_output','detections'))
                or row.get('retained_output_exact') is not (True if i else None)
                or plan.get('cuda_launches')!=2 or plan.get('temporary_arrays')!=1
                or plan.get('stencil_implementation')!='specialized-direct-experimental'):
            raise ValueError('Missing numerical, inference, retention, or implementation checks')
    if seen!=expected:raise ValueError('Incomplete stencil matrix')
    return dict(status='coverage_verified',executions=len(seen),revision=report['revision'],engine_sha256=report['engine_sha256'],
                qualification='Recorded fixed inference checks, not independent recomputation, traces, performance, or full stream acceptance')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x') as file:json.dump(verify(json.loads(a.report.read_text())),file,indent=2)


if __name__=='__main__':main()
