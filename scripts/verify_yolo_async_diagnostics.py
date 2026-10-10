"""Audit serial async capture and declared post-drain allocator observations."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from verify_yolo_trace import verify as verify_trace
from yolo_async_benchmark import require_gate


def verify(root, trace, gate):
    root = Path(root)
    report = json.loads((root / 'results.json').read_text())
    require_gate(gate, report.get('revision'), report.get('engine_sha256'), report.get('dirty', True))
    if (report.get('status') != 'passed' or report.get('drain_status') != 'passed'
            or report.get('mode') != 'serial-async-diagnostics-v1'
            or (report.get('blocks'), report.get('samples_per_block'), report.get('warmup')) != (10, 100, 100)):
        raise ValueError('Require complete separate diagnostics protocol')
    names = ('astronaut-1080p', 'chelsea-1080p')
    fixtures = report.get('fixtures', [])
    if (tuple(f['name'] for f in fixtures) != names
            or any(f['checked_frames'] != 2000 or f['failed_frames'] != 0 for f in fixtures)):
        raise ValueError('Incomplete real-inference coverage')
    checkpoints = report.get('allocator_checkpoints', [])
    if [(c['fixture'], c['block']) for c in checkpoints] != [(n, b) for n in names for b in range(-1, 10)]:
        raise ValueError('Missing ordered post-drain checkpoints')
    fields = ('cupy_used_bytes', 'cupy_total_bytes', 'torch_allocated_bytes', 'torch_reserved_bytes')
    for c in checkpoints:
        if any(type(c.get(k)) is not int or c[k] < 0 for k in fields):
            raise ValueError('Invalid allocator observations')
        if c[fields[0]] > c[fields[1]] or c[fields[2]] > c[fields[3]]:
            raise ValueError('Allocated bytes exceed reserved bytes')
    samples = root / 'samples.csv'
    if not samples.exists():
        samples = root / 'samples.csv.gz'
    opener = gzip.open if samples.suffix == '.gz' else open
    with opener(samples, 'rt', newline='') as handle:
        rows = list(csv.DictReader(handle))
    identities = [(n, b, i, candidate) for n in names for b in range(10)
                  for candidate in (('sync', 'submit') if b % 2 == 0 else ('submit', 'sync'))
                  for i in range(100)]
    if len(rows) != len(identities):
        raise ValueError('Incomplete diagnostic raw frames')
    for row, expected in zip(rows, identities):
        if (row['fixture'], int(row['block']), int(row['sample']), row['candidate']) != expected or row['correct'] != 'True':
            raise ValueError('Invalid diagnostic raw frame')
    growth = {}
    for name in names:
        rows = [c for c in checkpoints if c['fixture'] == name]
        growth[name] = {k: max(c[k] for c in rows) - rows[0][k] for k in fields}
    capture = verify_trace(Path(trace), candidates=('sync', 'submit'))
    for spans in capture['ranges'].values():
        for span in spans:
            if any(c['cupti_kind'] in (1, 2) and c['largest_copy_bytes'] >= 2457600 for c in span['copies']):
                raise ValueError('Frame/tensor-sized host transfer in completed inference range')
    return {'status': 'diagnostics_verified', 'checked_frames': 4000, 'trace': capture,
            'samples_sha256': hashlib.sha256(gzip.decompress(samples.read_bytes()) if samples.suffix == '.gz' else samples.read_bytes()).hexdigest(),
            'post_drain_growth_bytes': growth,
            'allocator_selection': 'stable' if all(v == 0 for row in growth.values() for v in row.values()) else 'growth_observed',
            'qualification': 'Declared allocator checkpoints exclude TensorRT internal/driver memory and transient peaks; no overlap, latency, or long-soak acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--delayed-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from yolo_common import digest
    report = json.loads((args.root / 'results.json').read_text())
    if report.get('delayed_report_sha256') != digest(args.delayed_report):
        raise ValueError('Delayed gate file hash mismatch')
    result = verify(args.root, args.trace, json.loads(args.delayed_report.read_text()))
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
