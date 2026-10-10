"""Retain sanitized selected inference ranges, not a whole-process trace."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from verify_yolo_trace import verify


def project(source, target, candidates):
    source, target = Path(source), Path(target)
    if target.exists(): raise ValueError('Projection must use a new path')
    original = verify(source, candidates=candidates)
    labels = ('known_4096_byte_d2h_control',) + tuple('yolo_' + c + '_complete' for c in candidates)
    with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'TARGET_INFO_SYSTEM_ENV' in tables and db.execute('SELECT COUNT(*) FROM TARGET_INFO_SYSTEM_ENV').fetchone()[0]:
            raise ValueError('Sanitize environment rows before projection')
        if any(re.search(r'rpa_[A-Za-z0-9_-]+', row[0]) for row in db.execute('SELECT value FROM StringIds')):
            raise ValueError('Sanitize credential-bearing strings before projection')
        placeholders = ','.join('?' for _ in labels)
        boundaries = db.execute('SELECT start,end FROM NVTX_EVENTS WHERE text IN (' + placeholders + ')', labels).fetchall()
        overlap = ' OR '.join('(start<? AND end>?)' for _ in boundaries)
        parameters = tuple(value for start, end in boundaries for value in (end, start))
        copied = {}
        with sqlite3.connect(target) as out:
            for table in ('NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'StringIds'):
                if table not in tables:
                    if table == 'CUPTI_ACTIVITY_KIND_RUNTIME': continue
                    raise ValueError('Missing trace table: ' + table)
                out.execute(db.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?', ('table', table)).fetchone()[0])
                if table == 'StringIds': rows = db.execute('SELECT * FROM StringIds').fetchall()
                elif table == 'NVTX_EVENTS': rows = db.execute('SELECT * FROM NVTX_EVENTS WHERE text IN (' + placeholders + ')', labels).fetchall()
                else: rows = db.execute('SELECT * FROM ' + table + ' WHERE ' + overlap, parameters).fetchall()
                if rows:
                    out.executemany('INSERT INTO ' + table + ' VALUES(' + ','.join('?' for _ in rows[0]) + ')', rows)
                copied[table] = len(rows)
    projected = verify(target, candidates=candidates)
    if {k: v for k, v in original.items() if k != 'trace_sha256'} != {k: v for k, v in projected.items() if k != 'trace_sha256'}:
        raise ValueError('Projection changed selected range observations')
    return dict(status='projection_verified', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                projection_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), rows=copied,
                candidates=list(candidates), scope='Selected completed inference ranges and known transfer control only. Not a whole-process trace.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', action='append', required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    args = parser.parse_args()
    report = project(args.source, args.output, tuple(args.candidate))
    with args.provenance.open('x') as handle: json.dump(report, handle, indent=2)


if __name__ == '__main__': main()
