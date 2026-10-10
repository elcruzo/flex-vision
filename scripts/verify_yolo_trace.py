"""Verify trace capture with a known transfer and report complete YOLO ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def verify(path, candidates=('cpg', 'torch', 'cvcuda')):
    result = {'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'scope':'completed resident-input preprocessing, TensorRT, and CUDA NMS ranges; excludes uploads and validation downloads',
              'ranges':{}}
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        control = db.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=?',
                             ('known_4096_byte_d2h_control',)).fetchall()
        if len(control) != 1 or control[0][1] is None:
            raise ValueError('Missing completed transfer control')
        start,end = control[0]
        copies = db.execute('SELECT copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?',
                            (end,start)).fetchall()
        if copies != [(2,4096)]:
            raise ValueError('Known 4096-byte device-to-host transfer was not captured exactly')
        result['known_control'] = {'kind':2,'bytes':4096,'captured':True}
        for candidate in candidates:
            spans = db.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=? ORDER BY start',
                               ('yolo_'+candidate+'_complete',)).fetchall()
            if len(spans) != 2:
                raise ValueError('Missing complete fixture ranges for '+candidate)
            result['ranges'][candidate] = []
            for start,end in spans:
                if end is None or end <= start:
                    raise ValueError('Incomplete candidate range')
                kernels = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?',
                                     (start,end)).fetchone()[0]
                if kernels <= 0:
                    raise ValueError('No GPU work captured')
                copies = db.execute('SELECT copyKind,COUNT(*),SUM(bytes),MAX(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>? GROUP BY copyKind',
                                    (end,start)).fetchall()
                result['ranges'][candidate].append({'kernel_launches':kernels,
                    'copies':[{'cupti_kind':kind,'count':count,'bytes':size,'largest_copy_bytes':largest}
                              for kind,count,size,largest in copies]})
    result['status'] = 'capture_verified'
    result['qualification'] = 'Copy counts are observations. Small metadata transfers must not be reported as zero host copies.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = verify(args.trace)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2)
        file.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
