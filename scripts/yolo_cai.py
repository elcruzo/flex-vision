"""Real TensorRT checks for CAI-only producers and bidirectional ordering."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import weakref


class Exporter:
    """Retain the allocation and advertised native producer stream."""
    def __init__(self, array, stream, handle):
        self.array, self.stream = array, stream
        self.__cuda_array_interface__ = dict(array.__cuda_array_interface__, version=3, stream=handle)


def require_pending(event, boundary):
    if event.done:
        raise AssertionError(boundary + ' completed before its pending observation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Output already exists')
    import cupy as cp
    import numpy as np
    import torch
    from cpg.reference import numpy_reference
    from yolo_common import pipeline, cases, digest, decode, compare_dense, compare_detections
    from pending_yolo_consumer import PendingConsumer
    report = dict(status='failed', mode='cai-v2',
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
        engine_sha256=digest(args.engine), delay_cycles=500000000, preprocessing_delay_cycles=1000000000, checks=[],
        scope='CAI-only input correctness and ownership; artificial producer delay; no performance claim')
    producer = cp.cuda.Stream(non_blocking=True)
    pre = cp.cuda.Stream(non_blocking=True)
    downstream = cp.cuda.Stream(non_blocking=True)
    streams = [producer, pre, downstream, cp.cuda.Stream.null, cp.cuda.Stream.ptds]
    delay = cp.RawKernel('extern "C" __global__ void delay(unsigned long long n) { unsigned long long t=clock64(); while(clock64()-t<n){} }','delay')
    pending = job = retained = None
    try:
        if report['dirty']: raise ValueError('Require clean committed source')
        consumer = PendingConsumer(args.engine, downstream)
        pipe = pipeline()
        with producer: delay((1,), (1,), (np.uint64(1),))
        producer.synchronize()
        for name, rgb, _, _ in cases():
            host = np.ascontiguousarray(rgb[...,::-1])
            expected, _ = numpy_reference(pipe, host)
            with np.load(args.controls / (name+'-controls.npz')) as controls:
                np.testing.assert_array_equal(expected, controls['tensor'])
                with torch.cuda.stream(consumer._torch_stream):
                    reference = torch.from_numpy(controls['tensorrt_dense']).cuda()
            downstream.synchronize()
            # Warm this shape's kernel before a deliberate pending-producer observation.
            with pre:
                warm_source = cp.asarray(host)
                warm_image = pipe(warm_source)
            warm = consumer(warm_image)
            with torch.cuda.stream(consumer._torch_stream):
                compare_dense(reference, warm, strict_output=True)
            del warm_source, warm_image, warm
            for layout in ('contiguous','reverse_rows','reverse_channels'):
                storage_host = host if layout=='contiguous' else host[::-1] if layout=='reverse_rows' else host[...,::-1]
                for label, stream, handle in [('explicit',producer,producer.ptr),('legacy',cp.cuda.Stream.null,1),('ptds',cp.cuda.Stream.ptds,2),('ready',producer,None)]:
                    for mode in ('sync','submit'):
                        report['active_scenario']=dict(fixture=name,layout=layout,producer=label,execution=mode)
                        report['stage']='producer_setup'
                        with stream:
                            uploaded = cp.asarray(np.ascontiguousarray(storage_host))
                            storage = cp.zeros_like(uploaded)
                        stream.synchronize()
                        view = storage if layout=='contiguous' else storage[::-1] if layout=='reverse_rows' else storage[...,::-1]
                        frame = Exporter(view,stream,handle)
                        owner = weakref.ref(frame)
                        with stream:
                            delay((1,), (1,), (np.uint64(report['delay_cycles']),))
                            cp.copyto(storage,uploaded)
                            ready = cp.cuda.Event(disable_timing=True)
                            ready.record(stream)
                        require_pending(ready, 'Producer')
                        if label=='ready': ready.synchronize()
                        with pre:
                            if mode=='submit':
                                delay((1,), (1,), (np.uint64(1000000000),))
                            if mode=='sync': image = pipe(frame)
                            else:
                                pending = pipe.submit(frame,stream=pre)
                                image = pending.wait_on(downstream)
                                require_pending(pending._event, 'Preprocessing')
                        del frame, view
                        gc.collect()
                        if mode=='submit' and owner() is None:
                            raise AssertionError('Exporter released before explicit completion')
                        report['stage']='producer_reuse'
                        # With no advertised stream, the caller orders future writes.
                        if mode=='submit' and label=='ready':
                            pending.wait()
                        pending_at_reuse = None
                        if mode=='submit' and label!='ready':
                            require_pending(pending._event, 'Preprocessing before producer reuse')
                            pending_at_reuse = True
                        with stream: storage.fill(0)
                        if mode=='submit':
                            pending.wait()
                            pending = None
                        else:
                            downstream.wait_event(pre.record())
                        report['stage']='tensor_and_inference_checks'
                        if retained is not None:
                            np.testing.assert_array_equal(cp.asnumpy(retained[0]).view(np.uint16),retained[1].view(np.uint16))
                        np.testing.assert_array_equal(cp.asnumpy(image).view(np.uint16),expected.view(np.uint16))
                        job = consumer.enqueue(image)
                        head = job.wait()
                        job = None
                        with torch.cuda.stream(consumer._torch_stream):
                            compare_dense(reference,head,strict_output=True)
                            compare_detections(decode(reference),decode(head))
                        stream.synchronize()
                        assert not cp.count_nonzero(storage).item()
                        report['checks'].append(dict(fixture=name,layout=layout,producer=label,execution=mode,
                            reuse_ordering='caller_completion' if label=='ready' else 'advertised_stream_fence',
                            pending_at_reuse=pending_at_reuse,
                            pending_producer=label!='ready',preprocessing_pending=True if mode=='submit' else None,
                            exporter_retained=True if mode=='submit' else None,
                            retained_output_exact=True if retained is not None else None,
                            producer_reuse=True,tensor_exact=True,dense_output=True,detections=True))
                        retained=(image,expected)
                        del image,head,storage,uploaded
            print('cai_validated',name,flush=True)
        report['active_scenario']=None
        report['stage']='complete'
        report['status']='passed'
    except Exception as exc:
        report['error']=repr(exc)
        raise
    finally:
        try:
            if pending is not None: pending.close()
            if job is not None: job.close()
            for stream in streams: stream.synchronize()
            report['drain_status']='passed'
        except Exception as exc:
            report['status']='failed';report['drain_status']='failed';report['drain_error']=repr(exc)
            raise
        finally:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
