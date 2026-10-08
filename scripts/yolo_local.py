"""Real YOLO inference through independent preprocessing paths on the Mac CPU."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from cpg.reference import numpy_reference, torch_reference
from yolo_common import (Dense, LIMITS, WEIGHTS_SHA256, load_model, pipeline, cases,
                         decode, detection_record, compare_detections, compare_dense, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('.cache/models/yolov8n.pt'))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'failed','scope':'CPU reference inference; no CUDA or TensorRT claim',
              'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'limits':LIMITS,'weights_sha256':WEIGHTS_SHA256,
              'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
              'sources':{str(p):digest(p) for p in [Path(__file__),Path('scripts/yolo_common.py'),*Path('src/cpg').glob('*.py')]},
              'config_sha256':digest('examples/yolo.yaml'),'cases':[]}
    try:
        torch.set_num_threads(4)
        model = Dense(load_model(args.weights)).eval()
        model_half = copy.deepcopy(model).half()
        pipe = pipeline()
        with torch.inference_mode():
            for name,rgb,expectation,provenance in cases():
                frame = rgb[...,::-1]
                expected,g = numpy_reference(pipe,frame)
                actual,_ = torch_reference(pipe,frame)
                ratio = min(640/frame.shape[0],640/frame.shape[1])
                if (round(frame.shape[1]*ratio),round(frame.shape[0]*ratio)) != (g.resized_width,g.resized_height):
                    raise ValueError('Native letterbox geometry differs; use its separate coordinate transform')
                native = LetterBox((640,640),auto=False)(image=np.ascontiguousarray(frame))
                native = torch.from_numpy(np.ascontiguousarray(native[...,::-1].transpose(2,0,1)[None])).float()/255
                native = native.half().float()
                case = {'id':name,'status':'failed','provenance':provenance}
                report['cases'].append(case)
                case['tensor_max_abs_error'] = float(np.abs(expected.astype(np.float32)-actual.numpy().astype(np.float32)).max())
                np.testing.assert_allclose(actual.numpy(),expected,atol=LIMITS['tensor_atol'],rtol=0)
                reference_head = model(torch.from_numpy(expected).float())
                actual_head = model(actual.float())
                native_head = model(native)
                case['dense'] = compare_dense(reference_head,actual_head)
                half_head = model_half(torch.from_numpy(expected))
                case['fp16_model_dense'] = compare_dense(reference_head,half_head)
                case['fp16_model_detection_ious'] = compare_detections(decode(reference_head),decode(half_head))
                case['fp16_model'] = detection_record(decode(half_head),g,expectation)
                reference_detection,actual_detection = decode(reference_head),decode(actual_head)
                case['matched_detection_ious'] = compare_detections(reference_detection,actual_detection)
                case['reference'] = detection_record(reference_detection,g,expectation)
                case['candidate'] = detection_record(actual_detection,g,expectation)
                case['native_ultralytics'] = detection_record(decode(native_head),g,expectation)
                case['native_tensor_max_abs_difference'] = float((native-torch.from_numpy(expected).float()).abs().max())
                np.savez_compressed(args.output/(name+'.npz'),expected=expected,actual=actual.numpy(),
                                    reference_dense=reference_head.numpy(),actual_dense=actual_head.numpy(),native_dense=native_head.numpy())
                case['status'] = 'passed'
                print(name,case['tensor_max_abs_error'],case['candidate']['expected_object_iou'],flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))


if __name__ == '__main__':
    main()
