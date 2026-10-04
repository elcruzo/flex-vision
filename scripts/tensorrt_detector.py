"""Run CPG output through a real TensorRT SSDLite network and CUDA postprocessing.

TensorRT executes trained normalization, backbone, and dense detection heads.
TorchVision decodes boxes and applies NMS on CUDA. This is a hybrid detector.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cupy as cp
import numpy as np
import onnx
import tensorrt as trt
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from cpg import load_pipeline
from cpg.reference import numpy_reference
from cpg.validation import photo_cases, check_expected_object
from local_detector import WEIGHTS_SHA256, digest


class DenseDetector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        transformed, _ = self.model.transform([image[0]])
        features = list(self.model.backbone(transformed.tensors).values())
        outputs = self.model.head(features)
        return outputs['bbox_regression'], outputs['cls_logits']

    def anchors(self, image):
        transformed, _ = self.model.transform([image[0]])
        features = list(self.model.backbone(transformed.tensors).values())
        return self.model.anchor_generator(transformed, features)


class Consumer:
    """Fixed FP32 test consumer. Owns runtime, engine, and execution context."""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT engine deserialization failed')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT context creation failed')
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_dtype(name) != trt.float32 or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError('This consumer requires FP32 device I/O')
            shape = tuple(self.engine.get_tensor_shape(name))
            if min(shape) <= 0:
                raise ValueError('This consumer requires fixed positive shapes')
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name != 'image' or shape != (1,3,320,320):
                    raise ValueError('Unexpected TensorRT input contract')
            else:
                self.outputs[name] = shape
        if set(self.outputs) != {'bbox_regression','cls_logits'}:
            raise ValueError('Unexpected TensorRT outputs')

    def __call__(self, image):
        if not isinstance(image, cp.ndarray) or image.shape != (1,3,320,320) or image.dtype != cp.float32 or not image.flags.c_contiguous:
            raise ValueError('Expected a contiguous CuPy FP32 NCHW 320x320 tensor')
        if image.device.id != torch.cuda.current_device():
            raise ValueError('Input must use the current CUDA device')
        outputs = {name: torch.empty(shape, device='cuda', dtype=torch.float32) for name,shape in self.outputs.items()}
        if not self.context.set_tensor_address('image', image.data.ptr):
            raise RuntimeError('Could not bind the CPG device pointer')
        for name, tensor in outputs.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f'Could not bind output {name}')
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()
        return outputs


def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    engine_path.write_bytes(bytes(serialized))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weights', type=Path, default=Path('.cache/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'))
    args = parser.parse_args()
    if digest(args.weights) != WEIGHTS_SHA256:
        parser.error('Model weights do not match the pinned hash')
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'failed','scope':'fixed FP32 TensorRT dense detector with TorchVision CUDA box decode/NMS',
              'torch':torch.__version__,'torchvision':torchvision.__version__,'tensorrt':trt.__version__,
              'cupy':cp.__version__,'onnx':onnx.__version__,'gpu':torch.cuda.get_device_name(0),
              'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
              'runner_sha256':digest(Path(__file__)),'weights_sha256':WEIGHTS_SHA256,
              'source_files':{str(p):digest(p) for p in sorted(Path('src/cpg').glob('*.py'))},
              'config_sha256':digest(Path('examples/local-detector.yaml')),
              'fixture_manifest_sha256':digest(Path('tests/fixtures/images/manifest.json')),
              'dense_atol':.01,'dense_rtol':1e-4,'score_atol':.001,'box_atol':.1,'cases':[]}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        model = ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None).eval()
        model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
        dense = DenseDetector(model).eval()
        sample = torch.zeros((1,3,320,320))
        onnx_path, engine_path = args.output/'ssdlite.onnx', args.output/'ssdlite.engine'
        # Fixed image contract. Decode/NMS remain outside the ONNX network.
        with torch.no_grad():
            torch.onnx.export(dense, sample, str(onnx_path), input_names=['image'],
                              output_names=['bbox_regression','cls_logits'],opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(onnx_path))
        started = time.perf_counter()
        build_engine(onnx_path,engine_path)
        report['build_seconds'] = time.perf_counter()-started
        report['onnx_sha256'],report['engine_sha256'] = digest(onnx_path),digest(engine_path)
        consumer = Consumer(engine_path)
        model.cuda()
        pipeline = load_pipeline('examples/local-detector.yaml')
        norm = pipeline.operations[1]
        if (pipeline.input_encoding != 'bgr8' or pipeline.plan((1080,1920,3))['output_shape'] != [1,3,320,320]
                or pipeline.operations[2].dtype != 'float32' or norm.mean != (0.,0.,0.) or norm.std != (1.,1.,1.) or norm.scale != 1/255):
            raise ValueError('This detector requires the fixed SSDLite fixture configuration')
        # Warm initialization lies outside each camera-to-network range.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), cp.cuda.ExternalStream(stream.cuda_stream, device_id=0), torch.inference_mode():
            warmup = cp.zeros((1,3,320,320),dtype=cp.float32)
            cp.cuda.get_current_stream().synchronize()
            consumer(warmup)
            anchors = dense.anchors(sample.cuda())
            torch.cuda.synchronize()
            # Nsight --capture-range=cudaProfilerApi excludes engine construction.
            torch.cuda.profiler.start()
            for case_id,rgb,expectation,provenance in photo_cases(Path('tests/fixtures/images/manifest.json')):
                host = np.ascontiguousarray(rgb[...,::-1])
                expected,g = numpy_reference(pipeline,host)
                frame = cp.asarray(host)
                cp.cuda.get_current_stream().synchronize()
                with torch.cuda.nvtx.range('cpg_to_tensorrt'):
                    with torch.cuda.nvtx.range('cpg_preprocess'):
                        actual = pipeline(frame)
                    head = consumer(actual)
                # Standard trained decoder and NMS execute on CUDA after TensorRT.
                candidate = model.postprocess_detections(head,anchors,[(320,320)])[0]
                baseline_tensor = torch.from_numpy(expected).cuda()
                expected_dense = dense(baseline_tensor)
                baseline = model([baseline_tensor[0]])[0]
                case = {'id':case_id,'provenance':provenance,'status':'failed','dense_max_abs_error':{}}
                report['cases'].append(case)
                for key,reference in zip(('bbox_regression','cls_logits'),expected_dense):
                    case['dense_max_abs_error'][key] = float((head[key]-reference).abs().max().item())
                    torch.testing.assert_close(head[key],reference,atol=.01,rtol=1e-4)
                actual_cpu = cp.asnumpy(actual)
                np.testing.assert_allclose(actual_cpu,expected,atol=2e-4,rtol=0)
                case['tensor_max_abs_error'] = float(np.abs(actual_cpu-expected).max())
                candidate = {k:v.cpu() for k,v in candidate.items()}
                baseline = {k:v.cpu() for k,v in baseline.items()}
                keep,other = candidate['scores']>=.5,baseline['scores']>=.5
                for key,atol in [('labels',0),('scores',.001),('boxes',.1)]:
                    torch.testing.assert_close(candidate[key][keep],baseline[key][other],atol=atol,rtol=0)
                boxes = [g.source_box(b.tolist()) for b in candidate['boxes'][keep]]
                case['expected_object_iou'] = check_expected_object(candidate['labels'][keep].tolist(),candidate['scores'][keep].tolist(),boxes,expectation)
                np.savez_compressed(args.output/f'{case_id}.npz',actual=actual_cpu,expected=expected,
                                    **{'candidate_'+k:v.numpy() for k,v in candidate.items()},
                                    **{'baseline_'+k:v.numpy() for k,v in baseline.items()})
                case['status'] = 'passed'
            torch.cuda.profiler.stop()
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))


if __name__ == '__main__':
    main()
