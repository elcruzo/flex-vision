"""Audit CAI-only complete ranges with matching real-inference correctness."""
import argparse
import json
from pathlib import Path
import sqlite3
from verify_yolo_cai import verify as verify_gate
from verify_yolo_trace import verify as verify_capture


PRODUCERS = ('explicit', 'legacy', 'ptds', 'ready')
EXECUTIONS = ('sync', 'submit')
FIXTURES = ('astronaut-1080p', 'chelsea-1080p')


def verify(report, gate, trace):
    accepted = verify_gate(gate)
    if (report.get('status') != 'passed' or report.get('drain_status') != 'passed'
            or report.get('dirty') is not False or report.get('mode') != 'cai-trace-v1'
            or report.get('revision') != accepted['revision']
            or report.get('engine_sha256') != accepted['engine_sha256']):
        raise ValueError('Require matching clean CAI capture and completed correctness gate')
    expected = {(f, p, e) for f in FIXTURES for p in PRODUCERS for e in EXECUTIONS}
    seen = set()
    for row in report.get('checks', []):
        identity = tuple(row.get(k) for k in ('fixture', 'producer', 'execution'))
        if identity not in expected or identity in seen or any(row.get(k) is not True for k in ('tensor_exact', 'dense_output', 'detections')):
            raise ValueError('Missing or duplicate CAI capture inference checks')
        seen.add(identity)
    if seen != expected: raise ValueError('Incomplete CAI capture matrix')
    candidates = tuple('cai_' + p + '_' + e for p in PRODUCERS for e in EXECUTIONS)
    capture = verify_capture(Path(trace), candidates=candidates)
    with sqlite3.connect(Path(trace).resolve().as_uri() + '?mode=ro', uri=True) as db:
        for candidate, spans in capture['ranges'].items():
            boundaries = db.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=? ORDER BY start',
                                    ('yolo_' + candidate + '_complete',)).fetchall()
            for (start, end), span in zip(boundaries, spans):
                count = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id '
                                   'WHERE k.start>=? AND k.end<=? AND s.value=?', (start, end, 'preprocess')).fetchone()[0]
                if count != 1 or span['kernel_launches'] <= 1:
                    raise ValueError('Require one CPG preprocessing kernel and downstream GPU work in each completed range')
                span['preprocess_launches'] = count
                host_copies = [c for c in span['copies'] if c['cupti_kind'] in (1, 2)]
                if sum(c['bytes'] for c in host_copies) >= 2457600:
                    raise ValueError('Tensor-sized aggregate host transfer in CAI complete range')
    return dict(status='cai_capture_verified', checked_frames=16, revision=accepted['revision'],
                engine_sha256=accepted['engine_sha256'], trace=capture,
                qualification='Fixed completed ranges exclude uploads and validation downloads. Small metadata copies remain reported. No latency, overlap, invalid-device, or external-library acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--cai-report', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from yolo_common import digest
    report = json.loads(args.report.read_text())
    if report.get('cai_report_sha256') != digest(args.cai_report):
        raise ValueError('CAI gate file hash mismatch')
    result = verify(report, json.loads(args.cai_report.read_text()), args.trace)
    with args.output.open('x') as handle: json.dump(result, handle, indent=2)


if __name__ == '__main__': main()
