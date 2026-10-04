"""Check synchronous CPG preprocessing ranges in an Nsight SQLite export.

This checks launch and transfer events only. It is not a latency benchmark.
The range must contain stream completion, as the current backend guarantees.
"""
import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--expected-ranges', type=int, required=True)
    parser.add_argument('--stage', choices=('preprocessing', 'tensorrt'), default='preprocessing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.expected_ranges < 1:
        parser.error('expected-ranges must be positive')
    range_name = 'cpg_preprocess' if args.stage == 'preprocessing' else 'cpg_to_tensorrt'
    report = {'status': 'failed', 'scope': f'synchronous {range_name} ranges only', 'ranges': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as dest:
        try:
            with sqlite3.connect(args.trace.resolve().as_uri()+'?mode=ro', uri=True) as db:
                ranges = db.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=? ORDER BY start', (range_name,)).fetchall()
                if len(ranges) != args.expected_ranges:
                    raise AssertionError(f'Expected {args.expected_ranges} ranges, found {len(ranges)}')
                for start, end in ranges:
                    if end is None or end <= start:
                        raise AssertionError('Preprocessing range is incomplete')
                    kernels = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?', (start,end)).fetchone()[0]
                    count, size = db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?', (end,start)).fetchone()
                    report['ranges'].append({'kernels': kernels, 'overlapping_memcpy_events': count, 'overlapping_memcpy_bytes': size})
                    valid_kernels = kernels == 1 if args.stage == 'preprocessing' else kernels > 1
                    if not valid_kernels or count != 0:
                        raise AssertionError('Unexpected kernel count or memory transfer in the selected range')
                report['status'] = 'passed'
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            json.dump(report,dest,indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
