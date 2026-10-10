"""Reject incomplete delayed-consumer inference evidence."""
import argparse
import json
from pathlib import Path
import re
from verify_yolo_stream_ownership import FIXTURES


def verify(report):
    if report.get('status') != 'passed' or report.get('dirty') is not False or report.get('drain_status') != 'passed':
        raise ValueError('Require a clean passing report and successful shutdown')
    if not re.fullmatch('[a-f0-9]{40}', report.get('revision','')) or not re.fullmatch('[a-f0-9]{64}', report.get('engine_sha256','')):
        raise ValueError('Require source and engine identity')
    if report.get('delay_cycles') != 500000000:
        raise ValueError('Require the fixed finite correctness delay')
    seen = set()
    for case in report.get('cases',[]):
        if case.get('name') not in FIXTURES or case['name'] in seen or case.get('status') != 'passed':
            raise ValueError('Unexpected, duplicate, or failed fixture')
        seen.add(case['name'])
        cycles = set()
        for row in case.get('checks',[]):
            key = (row.get('input'),row.get('cycle'))
            if key not in {(kind, index) for kind in ('cupy','torch') for index in range(4)} or key in cycles:
                raise ValueError('Missing or duplicate input cycle')
            cycles.add(key)
            if any(row.get(field) is not True for field in ('pending_after_close','rebind_rejected','dense_output','detections','input_tensor_exact','subsequent_tensor_exact')):
                raise ValueError('Incomplete pending-consumer checks')
        if len(cycles) != 8: raise ValueError('Incomplete fixture coverage')
    if seen != FIXTURES: raise ValueError('Incomplete fixture set')
    return {'status':'coverage_verified','executions':48,'revision':report['revision'],
            'engine_sha256':report['engine_sha256'],'qualification':'Recorded pending-consumer correctness; not independent tensor recomputation or performance acceptance'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Output report already exists')
    result=verify(json.loads(args.report.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__': main()
