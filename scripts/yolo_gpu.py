"""Fixed FP16 YOLO TensorRT correctness, matched timing, and transfer experiment."""
import argparse
import csv
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import cupy as cp
import cvcuda
import numpy as np
import onnx
import tensorrt as trt
import torch
import torch.nn.functional as F
from cpg.pipeline import Geometry
from cpg.reference import numpy_reference
from yolo_common import (Dense, LIMITS, WEIGHTS_SHA256, load_model, pipeline, cases,
                         decode, detection_record, compare_detections, compare_dense, digest)
from tensorrt_detector import build_engine


class Consumer:
    """Serial fixed-shape consumer. Each call returns a distinct owned output."""
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Execution context creation failed')
        self.io = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.io[name] = {'shape':list(self.engine.get_tensor_shape(name)),
                             'dtype':str(self.engine.get_tensor_dtype(name)),
                             'mode':str(self.engine.get_tensor_mode(name))}
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError('All I/O must reside on the GPU')
        if set(self.io) != {'image','predictions'}:
            raise ValueError('Unexpected engine I/O')
        if (self.io['image']['shape'] != [1,3,640,640]
                or self.engine.get_tensor_dtype('image') != trt.float16
                or self.engine.get_tensor_mode('image') != trt.TensorIOMode.INPUT
                or self.io['predictions']['shape'] != [1,84,8400]
                or self.engine.get_tensor_mode('predictions') != trt.TensorIOMode.OUTPUT):
            raise ValueError('Unexpected engine contract')
        output_type = self.engine.get_tensor_dtype('predictions')
        if output_type not in (trt.float16,trt.float32):
            raise ValueError('Unsupported output type')
        self.dtype = torch.float16 if output_type == trt.float16 else torch.float32

    def __call__(self, image):
        if (not isinstance(image,cp.ndarray) or image.shape != (1,3,640,640)
                or image.dtype != cp.float16 or not image.flags.c_contiguous
                or image.device.id != torch.cuda.current_device()):
            raise ValueError('Expected current-device contiguous FP16 1x3x640x640 CuPy input')
        result = torch.empty((1,84,8400),device='cuda',dtype=self.dtype)
        if not self.context.set_tensor_address('image',image.data.ptr):
            raise RuntimeError('Input pointer binding failed')
        if not self.context.set_tensor_address('predictions',result.data_ptr()):
            raise RuntimeError('Output pointer binding failed')
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()
        return result


def torch_preprocess(frame, g):
    image = torch.from_dlpack(frame).flip(-1).permute(2,0,1)[None].float()
    image = F.interpolate(image,size=(g.resized_height,g.resized_width),mode='bilinear',align_corners=False,antialias=False)
    image = F.pad(image,(g.left,640-g.resized_width-g.left,g.top,640-g.resized_height-g.top),value=114)
    image = (image*(1/255)).half().contiguous()
    torch.cuda.current_stream().synchronize()
    return cp.from_dlpack(image)


def vendor_preprocess(frame, g, stream):
    source = cvcuda.as_tensor(frame[None], 'NHWC')
    resized = cvcuda.resize_crop_convert_reformat(
        source,(g.resized_width,g.resized_height),cvcuda.Interp.LINEAR,
        cvcuda.RectI(0,0,g.resized_width,g.resized_height),layout='NCHW',
        data_type=cvcuda.Type.F16,manip=cvcuda.ChannelManip.REVERSE,
        scale=1/255,offset=0,srcCast=False,stream=stream)
    output = cp.empty((1,3,640,640),dtype=cp.float16)
    cvcuda.copymakeborder_into(cvcuda.as_tensor(output,'NCHW'),resized,cvcuda.Border.CONSTANT,[114/255]*3,
                             top=g.top,left=g.left,stream=stream)
    stream.sync()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('.cache/models/yolov8n.pt'))
    parser.add_argument('--engine',type=Path)
    parser.add_argument('--blocks',type=int,default=10)
    parser.add_argument('--samples',type=int,default=1000)
    parser.add_argument('--trace',action='store_true')
    args = parser.parse_args()
    if args.blocks <= 0 or args.samples <= 0:
        parser.error('blocks and samples must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'failed','limits':LIMITS,'acceptance_policy':'v3: task-relevant boxes, tighter same-engine scores; all background differences reported','weights_sha256':WEIGHTS_SHA256,'cases':[],
              'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
              'config_sha256':digest('examples/yolo.yaml'),
              'sources':{str(p):digest(p) for p in [Path(__file__),Path('scripts/yolo_common.py'),*Path('src/cpg').glob('*.py')]},
              'versions':{'torch':torch.__version__,'cupy':cp.__version__,'tensorrt':trt.__version__,
                          'cvcuda':cvcuda.__version__},'gpu':torch.cuda.get_device_name(0),
              'blocks':args.blocks,'samples_per_block':args.samples,'warmup':100,
              'unmeasured':['power','CPU utilization','GPU utilization','memory bandwidth','peak temporary memory'],
              'model_preparation':'FP32 convolution/batchnorm fusion before FP16 conversion',
              'vendor_plan':'resize/crop/convert/reformat to NCHW, then planar padding into owned output',
              'scope':'resident single-frame fixed FP16 dense TensorRT network with TorchVision CUDA NMS; synchronous fresh outputs'}
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_fp32 = Dense(load_model(args.weights)).cuda().eval()
        model = copy.deepcopy(model_fp32).half().eval()
        engine_path = args.engine or args.output/'yolo.engine'
        with torch.inference_mode():
            if args.engine is None:
                sample = torch.zeros((1,3,640,640),device='cuda',dtype=torch.float16)
                model(sample)
                onnx_path = args.output/'yolo.onnx'
                torch.onnx.export(model,sample,str(onnx_path),input_names=['image'],output_names=['predictions'],opset_version=17,dynamo=False)
                onnx.checker.check_model(onnx.load(onnx_path))
                build_engine(onnx_path,engine_path)
                report['onnx_sha256'] = digest(onnx_path)
            consumer = Consumer(engine_path)
            report['engine_io'] = consumer.io
            report['engine_sha256'] = digest(engine_path)
            (args.output/'engine-inspector.json').write_text(consumer.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
            pipe = pipeline()
            cvstream = cvcuda.Stream()
            stream = torch.cuda.ExternalStream(cvstream.handle)
            torch.cuda.synchronize()
            with torch.cuda.stream(stream),cp.cuda.ExternalStream(stream.cuda_stream,device_id=0):
                benchmark_cases = []
                for name,rgb,expectation,provenance in cases():
                    host = np.ascontiguousarray(rgb[...,::-1])
                    expected,g = numpy_reference(pipe,host)
                    frame = cp.asarray(host)
                    stream.synchronize()
                    reference_tensor = torch.from_numpy(expected).cuda()
                    reference_head = model(reference_tensor)
                    fp32_head = model_fp32(reference_tensor.float())
                    reference_detection = decode(reference_head)
                    reference_trt = consumer(cp.from_dlpack(reference_tensor))
                    case = {'id':name,'status':'failed','provenance':provenance,
                            'input_sha256':digest_bytes(host),'paths':{}}
                    report['cases'].append(case)
                    # Preserve dense values before assertions, including failing cases.
                    np.savez_compressed(args.output/(name+'-controls.npz'),tensor=expected,
                        fp32_dense=fp32_head.cpu().numpy(),fp16_dense=reference_head.cpu().numpy(),
                        tensorrt_dense=reference_trt.cpu().numpy())
                    case['fp16_vs_fp32_dense'] = compare_dense(fp32_head,reference_head)
                    compare_detections(decode(fp32_head),reference_detection)
                    case['same_input_trt_dense'] = compare_dense(reference_head,reference_trt,cross_engine=True)
                    compare_detections(reference_detection,decode(reference_trt))
                    runners = {'cpg':lambda:pipe(frame), 'torch':lambda:torch_preprocess(frame,g),
                               'cvcuda':lambda:vendor_preprocess(frame,g,cvstream)}
                    retained = []
                    for path,run in runners.items():
                        tensor = run()
                        downloaded = cp.asnumpy(tensor)
                        np.testing.assert_allclose(downloaded,expected,atol=LIMITS['tensor_atol'],rtol=0)
                        head = consumer(tensor)
                        detection = decode(head)
                        np.savez_compressed(args.output/(name+'-'+path+'.npz'),tensor=downloaded,dense=head.cpu().numpy())
                        case['paths'][path] = {'tensor_max_abs_error':float(np.abs(downloaded.astype(np.float32)-expected.astype(np.float32)).max()),
                            'dense':compare_dense(reference_trt,head,preprocessing=True),'matched_ious':compare_detections(decode(reference_trt),detection),
                            'detections':detection_record(detection,g,expectation)}
                        retained.append((tensor,downloaded.copy()))
                    # Actual inference also checks framework import and a negative channel stride.
                    for path,value in [('torch_dlpack',torch.from_dlpack(frame)),('negative_stride',cp.asarray(rgb)[...,::-1])]:
                        tensor = pipe(value)
                        np.testing.assert_allclose(cp.asnumpy(tensor),expected,atol=LIMITS['tensor_atol'],rtol=0)
                        case['paths'][path] = {'detections':detection_record(decode(consumer(tensor)),g,expectation)}
                    for tensor,snapshot in retained:
                        np.testing.assert_array_equal(cp.asnumpy(tensor),snapshot)
                    case['status'] = 'passed'
                    print('validated',name,flush=True)
                    if name.endswith('1080p'):
                        benchmark_cases.append((name,frame,g))
                if args.trace:
                    torch.cuda.profiler.start()
                    control = cp.arange(1024,dtype=cp.float32)
                    with torch.cuda.nvtx.range('known_4096_byte_d2h_control'):
                        cp.asnumpy(control)
                    for name,frame,g in benchmark_cases:
                        for path in ('cpg','torch','cvcuda'):
                            with torch.cuda.nvtx.range('yolo_'+path+'_complete'):
                                tensor = pipe(frame) if path == 'cpg' else torch_preprocess(frame,g) if path == 'torch' else vendor_preprocess(frame,g,cvstream)
                                decode(consumer(tensor))
                                stream.synchronize()
                    torch.cuda.profiler.stop()
                else:
                    with (args.output/'samples.csv').open('w',newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['fixture','block','sample','candidate','preprocess_ms','network_ms','nms_ms','complete_gpu_ms','complete_host_ms'])
                        for name,frame,g in benchmark_cases:
                            runners = {'cpg':lambda:pipe(frame),'torch':lambda:torch_preprocess(frame,g),
                                       'cvcuda':lambda:vendor_preprocess(frame,g,cvstream)}
                            for run in runners.values():
                                for _ in range(100):
                                    decode(consumer(run()))
                                stream.synchronize()
                            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                            names = list(runners)
                            for block in range(args.blocks):
                                for path in names[block%3:]+names[:block%3]:
                                    for index in range(args.samples):
                                        start = time.perf_counter_ns()
                                        events[0].record()
                                        tensor = runners[path]()
                                        events[1].record()
                                        head = consumer(tensor)
                                        events[2].record()
                                        detections = decode(head)
                                        events[3].record()
                                        stream.synchronize()
                                        host_ms = (time.perf_counter_ns()-start)/1e6
                                        writer.writerow([name,block,index,path,*[events[i].elapsed_time(events[i+1]) for i in range(3)],events[0].elapsed_time(events[3]),host_ms])
                                file.flush()
                                print('timed',name,block,flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))


def digest_bytes(array):
    import hashlib
    return hashlib.sha256(array.tobytes()).hexdigest()


if __name__ == '__main__':
    main()
