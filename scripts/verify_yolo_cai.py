"""Reject incomplete CAI-only real-inference reports."""
import argparse
import json
from pathlib import Path
import re
from verify_yolo_stream_ownership import FIXTURES


def verify(report):
    if (report.get('status')!='passed' or report.get('dirty') is not False
            or report.get('drain_status')!='passed' or report.get('mode')!='cai-v2'
            or report.get('delay_cycles')!=500000000 or report.get('preprocessing_delay_cycles')!=1000000000):
        raise ValueError('Require complete clean CAI protocol and drain')
    for field,length in [('revision',40),('engine_sha256',64)]:
        if not re.fullmatch('[a-f0-9]{'+str(length)+'}',report.get(field,'')):
            raise ValueError('Missing source or engine identity')
    expected={(f,l,p,e) for f in FIXTURES for l in ('contiguous','reverse_rows','reverse_channels')
              for p in ('explicit','legacy','ptds','ready') for e in ('sync','submit')}
    seen=set()
    for index,row in enumerate(report.get('checks',[])):
        identity=tuple(row.get(k) for k in ('fixture','layout','producer','execution'))
        if identity not in expected or identity in seen:
            raise ValueError('Unexpected or duplicate CAI scenario')
        seen.add(identity)
        if (row.get('reuse_ordering') != ('caller_completion' if row['producer']=='ready' else 'advertised_stream_fence')
                or row.get('pending_at_reuse') is not (True if row['execution']=='submit' and row['producer']!='ready' else None)
                or row.get('pending_producer') is not (row['producer']!='ready')
                or row.get('preprocessing_pending') is not (True if row['execution']=='submit' else None)
                or row.get('exporter_retained') is not (True if row['execution']=='submit' else None)
                or row.get('retained_output_exact') is not (True if index else None)
                or any(row.get(k) is not True for k in ('producer_reuse','tensor_exact','dense_output','detections'))):
            raise ValueError('Missing ordering, lifetime, or inference checks')
    if seen!=expected: raise ValueError('Incomplete CAI scenario matrix')
    return dict(status='coverage_verified',executions=len(seen),revision=report['revision'],
                engine_sha256=report['engine_sha256'],
                qualification='Recorded real-inference CAI coverage; not independent tensor recomputation, invalid-device acceptance, or transfer proof')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as handle:json.dump(verify(json.loads(args.report.read_text())),handle,indent=2)


if __name__=='__main__':main()
