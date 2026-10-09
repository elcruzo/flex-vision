"""Audit complete coverage of the synchronous detector stream scenario."""
import argparse
import json
from pathlib import Path
import re

FIXTURES = {'astronaut', 'astronaut-padded', 'chelsea', 'chelsea-padded', 'astronaut-1080p', 'chelsea-1080p'}


def verify(report, *, expected_mode="synchronous"):
    if report.get("mode", "synchronous") != expected_mode:
        raise ValueError("Unexpected execution mode")
    if report.get('status') != 'passed' or report.get('dirty') is not False:
        raise ValueError('Require a passing clean-source GPU report')
    if not re.fullmatch(r'[0-9a-f]{40}', report.get('revision', '')):
        raise ValueError('Require the full source revision')
    if not re.fullmatch(r'[0-9a-f]{64}', report.get('engine_sha256', '')):
        raise ValueError('Require the engine checksum')
    streams = report.get('stream_handles', [])
    if len(streams) != 3 or any(type(s) is not int or s <= 0 for s in streams) or len(set(streams)) != 3:
        raise ValueError('Require three distinct non-default stream handles')
    expected = {(name, kind) for name in FIXTURES for kind in ('cupy', 'torch')}
    seen = set()
    executions = 0
    for case in report.get('cases', []):
        key = (case.get('name'), case.get('input'))
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate fixture/input case')
        seen.add(key)
        rows = case.get('checks', [])
        if case.get('status') != 'passed' or case.get('retention') != 'passed' or len(rows) != 4:
            raise ValueError('Require four checked cycles and output retention')
        for cycle, row in enumerate(rows):
            if row.get('cycle') != cycle or any(row.get(field) is not True for field in
                    ('tensor_exact', 'dense_output', 'detections', 'retained_outputs', 'consumer_complete')):
                raise ValueError('Missing per-cycle inference or ownership check')
        if expected_mode == 'async-submit' and any(row.get('source_owner_release') is not True for row in rows):
            raise ValueError('Require early caller source-owner release')
        executions += len(rows)
    if seen != expected:
        raise ValueError('Incomplete fixture/input coverage')
    if report.get('drain_status') != 'passed':
        raise ValueError('Stream shutdown did not complete')
    return {'status': 'coverage_verified', 'executions': executions, 'cases': len(seen),
            'mode': expected_mode, 'revision': report['revision'], 'engine_sha256': report['engine_sha256'],
            'qualification': 'Audit of recorded GPU checks; not independent tensor recomputation or asynchronous acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('synchronous', 'async-submit'), default='synchronous')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output audit already exists')
    result = verify(json.loads(args.report.read_text()), expected_mode=args.mode)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
