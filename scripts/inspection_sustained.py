"""Exercise bounded resident camera streams through real classifier inference."""
import argparse
from collections import deque
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small

from cpg import InspectionPipeline
from inspection_reference import numpy_inspection, TorchInspectionBaseline
from local_classifier import cases, check_semantics, WEIGHTS_SHA256
from local_detector import digest


def run(model, sources, expected, expected_logits, roi, strategy, cameras, fps,
        seconds, depth, repeat, stream_pool):
    streams = stream_pool[:cameras]
    executors = []
    for stream in streams:
        with torch.cuda.stream(stream):
            executors.append(TorchInspectionBaseline('cuda') if strategy == 'baseline'
                             else InspectionPipeline(roi, 'roi').prepare(sources[0].shape))
    torch.cuda.synchronize()

    def enqueue(lane, variant):
        executor = executors[lane]
        tensor = (executor.from_bgr8(sources[variant], roi) if strategy == 'baseline'
                  else executor(sources[variant]))
        return tensor, model(tensor)

    with torch.inference_mode():
        for lane, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                for i in range(10):
                    tensor, logits = enqueue(lane, i % 2)
                    del tensor, logits
        torch.cuda.synchronize()
        total = cameras * fps * seconds
        # Fixed storage: one flag per offered frame, no retained image payloads.
        checks = torch.zeros(total, dtype=torch.bool, device='cuda')
        torch.cuda.synchronize()
        allocated_start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        queues = [deque() for _ in streams]
        rows, memory = [], []
        start = time.perf_counter()
        interval = 1 / (cameras * fps)
        next_frame = 0
        max_pending = 0
        next_memory = start
        while next_frame < total or any(queues):
            now = time.perf_counter()
            if now > start + seconds + 30:
                raise TimeoutError('Bounded workload did not drain within 30 seconds')
            for queue in queues:
                while queue and queue[0][1].query():
                    row, done, first, inferred = queue.popleft()
                    row['completion_s'] = time.perf_counter() - start
                    row['arrival_to_completion_ms'] = 1000 * (row['completion_s'] - row['arrival_s'])
                    row['gpu_pipeline_ms'] = first.elapsed_time(inferred)
                    rows.append(row)
            if now >= next_memory:
                memory.append({'elapsed_s': now-start, 'allocated': torch.cuda.memory_allocated(),
                               'reserved': torch.cuda.memory_reserved()})
                next_memory = now + .1
            if next_frame < total and now >= start + next_frame * interval:
                index = next_frame
                next_frame += 1
                lane = (index + repeat) % cameras
                variant = (index // cameras) % 2
                row = {'frame': index, 'lane': lane, 'variant': variant,
                       'arrival_s': index * interval, 'submission_s': now-start,
                       'status': 'completed'}
                # A missed arrival slot is dropped, never replayed as a burst.
                if now >= start + (index + 1) * interval:
                    row['status'] = 'dropped_dispatch_late'
                elif len(queues[lane]) >= depth:
                    row['status'] = 'dropped_queue_full'
                else:
                    with torch.cuda.stream(streams[lane]):
                        first = torch.cuda.Event(enable_timing=True)
                        inferred = torch.cuda.Event(enable_timing=True)
                        done = torch.cuda.Event()
                        first.record()
                        tensor, logits = enqueue(lane, variant)
                        inferred.record()
                        # GPU checks are equal for both paths. They remain inside load.
                        tensor_ok = (torch.abs(tensor-expected[variant]) <= 2e-4).all()
                        logit_ok = (torch.abs(logits-expected_logits[variant]) <=
                                    .001 + 1e-4 * torch.abs(expected_logits[variant])).all()
                        labels_ok = (logits.topk(5).indices == expected_logits[variant].topk(5).indices).all()
                        checks[index] = tensor_ok & logit_ok & labels_ok
                        done.record()
                    del tensor, logits, tensor_ok, logit_ok, labels_ok
                    queues[lane].append((row, done, first, inferred))
                    max_pending = max(max_pending, sum(map(len, queues)))
                    continue
                rows.append(row)
            else:
                time.sleep(.0001)
        elapsed = time.perf_counter() - start
        torch.cuda.synchronize()
        flags = checks.cpu().numpy()
    completed = [row for row in rows if row['status'] == 'completed']
    failures = [row['frame'] for row in completed if not flags[row['frame']]]
    latencies = [row['arrival_to_completion_ms'] for row in completed]
    report = {'strategy': strategy, 'cameras': cameras, 'fps_per_camera': fps,
              'seconds': seconds, 'repeat': repeat, 'queue_depth_per_camera': depth,
              'offered': total, 'completed': len(completed), 'dropped': total-len(completed),
              'drop_reasons': {reason: sum(row['status'] == reason for row in rows)
                               for reason in ('dropped_dispatch_late', 'dropped_queue_full')},
              'elapsed_with_drain_s': elapsed,
              'completed_per_second_with_drain': len(completed)/max(seconds, elapsed),
              'completed_within_window': sum(row['completion_s'] <= seconds for row in completed),
              'arrival_to_completion_ms': dict(zip(('p50','p95','p99'),
                  np.percentile(latencies, [50,95,99]).tolist())) if latencies else None,
              'max_inflight': max_pending, 'allocated_start': allocated_start,
              'allocated_after_drain': torch.cuda.memory_allocated(),
              'peak_allocated': torch.cuda.max_memory_allocated(),
              'peak_reserved': torch.cuda.max_memory_reserved(),
              'failed_frames': failures, 'memory_samples': memory,
              'status': 'passed' if completed and not failures and len(rows) == total
                        and max_pending <= cameras * depth else 'failed'}
    return report, sorted(rows, key=lambda row: row['frame'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--weights', type=Path, default=Path('.cache/models/mobilenet_v3_small-047dcff4.pth'))
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60 or not 1 <= args.repeats <= 8 or not 1 <= args.depth <= 4:
        parser.error('Require seconds 1..60, repeats 1..8, depth 1..4')
    if not torch.cuda.is_available():
        parser.error('CUDA is required; no CPU fallback')
    if digest(args.weights) != WEIGHTS_SHA256:
        parser.error('Require pinned MobileNetV3 weights')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True):
        parser.error('Commit the measured source first')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'runs': [], 'scope': 'resident synthetic arrivals with GPU validation overhead',
              'stream_policy': 'four persistent streams, shared across strategies and repeats',
              'revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'weights_sha256': WEIGHTS_SHA256,
              'source_sha256': {p: digest(Path(p)) for p in (
                  'scripts/inspection_sustained.py', 'scripts/inspection_reference.py',
                  'scripts/local_classifier.py', 'src/cpg/inspection.py',
                  'tests/fixtures/images/manifest.json')}}
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        cpu_model = mobilenet_v3_small(weights=None).eval()
        cpu_model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
        model = copy.deepcopy(cpu_model).cuda().eval()
        _, rgb, roi, _ = next(case for case in cases() if case[0] == 'chelsea-4k-roi')
        variants = [np.ascontiguousarray(rgb[...,::-1]), np.ascontiguousarray(rgb[:,::-1,::-1])]
        report['input_sha256'] = [hashlib.sha256(x.tobytes()).hexdigest() for x in variants]
        report['roi'] = list(roi)
        with torch.inference_mode():
            expected_cpu = [torch.from_numpy(numpy_inspection(x, roi)) for x in variants]
            logits_cpu = [cpu_model(x) for x in expected_cpu]
            for logits in logits_cpu:
                check_semantics(logits)
            expected = [x.cuda() for x in expected_cpu]
            expected_logits = [x.cuda() for x in logits_cpu]
            sources = [torch.from_numpy(x).cuda() for x in variants]
            torch.cuda.synchronize()
            stream_pool = [torch.cuda.Stream() for _ in range(4)]
            for cameras, fps in ((1,30), (4,60)):
                for repeat in range(args.repeats):
                    order = ('baseline','roi') if repeat % 2 == 0 else ('roi','baseline')
                    for strategy in order:
                        result, rows = run(model, sources, expected, expected_logits, roi, strategy,
                                           cameras, fps, args.seconds, args.depth, repeat, stream_pool)
                        filename = f'{cameras}cams-{strategy}-{repeat}.csv'
                        with (args.output/filename).open('w', newline='') as handle:
                            writer = csv.DictWriter(handle, fieldnames=['frame','lane','variant','arrival_s',
                                'submission_s','status','completion_s','arrival_to_completion_ms','gpu_pipeline_ms'])
                            writer.writeheader()
                            writer.writerows(rows)
                        result['samples'] = filename
                        report['runs'].append(result)
                        print(json.dumps({k: result[k] for k in ('strategy','cameras','repeat','completed','dropped','status')}), flush=True)
                        if result['status'] != 'passed':
                            raise AssertionError('Sustained inference validation failed')
        ratios = {1: [], 4: []}
        for cameras in ratios:
            for repeat in range(args.repeats):
                pair = {r['strategy']: r for r in report['runs']
                        if r['cameras'] == cameras and r['repeat'] == repeat}
                baseline, candidate = pair['baseline'], pair['roi']
                ratios[cameras].append(
                    candidate['completed_per_second_with_drain']/baseline['completed_per_second_with_drain']
                    if cameras == 4 else
                    candidate['arrival_to_completion_ms']['p99']/baseline['arrival_to_completion_ms']['p99'])
        throughput_ratio = float(np.median(ratios[4]))
        tail_ratio = float(np.median(ratios[1]))
        report['performance_hypothesis'] = {
            'stress_throughput_ratios': ratios[4], 'control_p99_ratios': ratios[1],
            'median_stress_throughput_ratio': throughput_ratio,
            'median_control_p99_ratio': tail_ratio,
            'passed': throughput_ratio >= 1.2 and tail_ratio <= 1.05}
        report['status'] = 'passed'  # Correctness only; performance has a separate verdict.
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
