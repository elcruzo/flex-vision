"""Count GPU work and copies inside completed synchronous NVTX ranges.

This reports observations. A memory copy is not automatically a failure:
postprocessing can synchronize small metadata without copying image payloads.
"""
import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--range',dest='range_name',required=True)
    parser.add_argument('--expected-ranges',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.expected_ranges < 1:
        parser.error('expected-ranges must be positive')
    report = {'range':args.range_name,'scope':'synchronous completed ranges; counts only, not performance timing','ranges':[]}
    with sqlite3.connect(args.trace.resolve().as_uri()+'?mode=ro',uri=True) as db:
        spans = db.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=? ORDER BY start',(args.range_name,)).fetchall()
        if len(spans) != args.expected_ranges:
            raise ValueError(f'Expected {args.expected_ranges} ranges, found {len(spans)}')
        for start,end in spans:
            if end is None or end <= start:
                raise ValueError('Incomplete range')
            kernels = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?',(start,end)).fetchone()[0]
            if kernels == 0:
                raise ValueError('No kernels in completed range')
            copies = db.execute('SELECT copyKind,COUNT(*),SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>? GROUP BY copyKind',(end,start)).fetchall()
            report['ranges'].append({'kernels':kernels,'copies_by_cupti_kind':[
                {'kind':kind,'count':count,'bytes':size} for kind,count,size in copies],
                'overlapping_copy_events':sum(row[1] for row in copies),
                'overlapping_copy_bytes':sum(row[2] for row in copies)})
    with args.output.open('x') as dest:
        json.dump(report,dest,indent=2)
        dest.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
