"""Check actual standard ROS tensor serialization and explicit CPU views."""
import json
from pathlib import Path
import numpy as np
from rclpy.serialization import serialize_message, deserialize_message
from tensor_msgs.msg import ExperimentalTensor
from cpg.ros import write_cpu_tensor, cpu_tensor_view


def main():
    checks=[]
    for dtype in ('float16','float32'):
        source=np.linspace(-3,3,1*3*224*224,dtype=np.float32).reshape(1,3,224,224).astype(dtype)
        encoded=write_cpu_tensor(ExperimentalTensor(),source)
        received=deserialize_message(serialize_message(encoded),ExperimentalTensor)
        borrowed=cpu_tensor_view(received,expected_shape=source.shape,expected_dtype=dtype)
        np.testing.assert_array_equal(borrowed.pixels.view(np.uint8),source.view(np.uint8))
        assert borrowed.owner is received and not borrowed.pixels.flags.writeable
        controls=[]
        for name in ('shape','dtype','strides','offset','truncated'):
            bad=deserialize_message(serialize_message(encoded),ExperimentalTensor)
            if name=='shape':bad.shape=[1,3,224,223]
            elif name=='dtype':bad.dtype_lanes=2
            elif name=='strides':bad.strides=[150528,50176,224,1]
            elif name=='offset':bad.byte_offset=2
            else:bad.data=bad.data[:-1]
            try:cpu_tensor_view(bad,expected_shape=source.shape,expected_dtype=dtype)
            except ValueError as exc:controls.append(dict(name=name,error=str(exc)))
            else:raise AssertionError('Invalid tensor accepted: '+name)
        checks.append(dict(dtype=dtype,shape=list(source.shape),payload_bytes=source.nbytes,
                           exact_bits=True,owner_retained=True,readonly=True,controls=controls))
    Path('/results/contract.json').write_text(json.dumps(dict(status='passed',checks=checks,
        scope='Actual CPU ROS serialization contract only; no transport, inference, GPU or camera metadata acceptance'),indent=2)+'\n')


if __name__=='__main__':main()
