"""Bounded CPU ROS Image -> explicit CPG reference -> real classifier gate."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=('publish','subscribe'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--negative-control',action='store_true',help='Require an absent dog class at subscribed-image inference; expected failure')
    args=parser.parse_args()
    if args.negative_control and args.role!='subscribe':parser.error('Negative control applies only to subscriber')
    if args.output.exists():raise ValueError('Result already exists')
    import numpy as np
    import torch
    import torchvision
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import Image
    from torchvision.models import mobilenet_v3_small
    from cpg import Pipeline
    from cpg.reference import numpy_reference,torch_reference
    from cpg.ros import cpu_image_view
    from cpg.validation import photo_cases
    from local_classifier import check_semantics,WEIGHTS_SHA256
    from local_detector import digest
    torch.set_num_threads(4)
    rgb=next(rgb for name,rgb,_,_ in photo_cases(Path('tests/fixtures/images/manifest.json')) if name=='chelsea')
    frame=np.ascontiguousarray(rgb[...,::-1]);h,w,_=frame.shape;step=w*3+17
    payload=np.full((h,step),114,dtype=np.uint8);payload[:,:w*3]=frame.reshape(h,w*3)
    report=dict(status='failed',role=args.role,revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,
                source_image_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),payload_sha256=hashlib.sha256(payload.tobytes()).hexdigest(),
                width=w,height=h,step=step,frames=0,checks=[],tensor_atol=2e-4,logit_atol=.001,logit_rtol=1e-4,
                negative_control=args.negative_control,scope='CPU Image transport, explicit CPU preprocessing and pinned CPU classifier; no GPU, live camera, tensor publication or performance acceptance')
    node=None
    try:
        if report['dirty']:raise ValueError('Require clean committed source')
        references={}
        if args.role=='subscribe':
            weights=Path('.cache/models/mobilenet_v3_small-047dcff4.pth')
            if digest(weights)!=WEIGHTS_SHA256:raise ValueError('Pinned model hash mismatch')
            report['weights_sha256']=WEIGHTS_SHA256
            model=mobilenet_v3_small(weights=None).eval()
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
            for size in (3,5,7,9):
                pipe=Pipeline(input_encoding='bgr8').gaussian(size,sigma=1.2).letterbox(224,224).normalize([.485,.456,.406],[.229,.224,.225],scale=1/255).to(dtype='float32',layout='NCHW')
                expected,_=numpy_reference(pipe,frame)
                with torch.inference_mode():baseline=model(torch.from_numpy(expected))
                labels,_=check_semantics(baseline)
                references[size]=(pipe,expected,baseline,labels)
        rclpy.init()
        class Gate(Node):
            def __init__(self):
                super().__init__('cpg_cpu_classifier_'+args.role)
                self.started=time.monotonic();self.count=0;self.drain=0
                qos=QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE)
                if args.role=='publish':self.pub=self.create_publisher(Image,'cpg_reference/image',qos)
                else:self.sub=self.create_subscription(Image,'cpg_reference/image',self.receive,qos)
                self.timer=self.create_timer(.5,self.tick)
            def finish(self):
                report.update(status='passed',frames=self.count);rclpy.shutdown()
            def tick(self):
                if time.monotonic()-self.started>45:raise TimeoutError('Bounded ROS inference deadline')
                if args.role!='publish':return
                if self.count==8:
                    self.drain+=1
                    if self.drain==2:self.finish()
                    return
                if not self.pub.get_subscription_count():
                    if self.count:raise RuntimeError('Consumer disappeared before completion')
                    return
                msg=Image();msg.width=w;msg.height=h;msg.step=step;msg.encoding='bgr8';msg.is_bigendian=0
                msg.header.frame_id='cpg_cat_'+str(self.count);msg.header.stamp=self.get_clock().now().to_msg()
                msg.data=array('B',payload.tobytes());self.pub.publish(msg);self.count+=1
            def receive(self,msg):
                if (msg.width,msg.height,msg.step,msg.encoding,msg.is_bigendian,msg.header.frame_id)!=(w,h,step,'bgr8',0,'cpg_cat_'+str(self.count)):
                    raise AssertionError('Image metadata or sequence mismatch')
                raw=np.frombuffer(msg.data,dtype=np.uint8)
                if raw.size!=h*step:raise AssertionError('Image payload size mismatch')
                np.testing.assert_array_equal(raw.reshape(h,step),payload)
                decoded=cpu_image_view(msg,expected_encoding='bgr8')
                view=decoded.pixels
                size=(3,5,7,9)[self.count%4];pipe,expected,baseline,labels=references[size]
                actual,_=torch_reference(pipe,view,device='cpu')
                np.testing.assert_allclose(actual.numpy(),expected,atol=report['tensor_atol'],rtol=0)
                with torch.inference_mode():head=model(actual)
                torch.testing.assert_close(head,baseline,atol=report['logit_atol'],rtol=report['logit_rtol'])
                actual_labels,mass=check_semantics(head,negative=args.negative_control)
                if actual_labels!=labels:raise AssertionError('Class ordering mismatch')
                report['checks'].append(dict(sequence=self.count,size=size,frame_id=msg.header.frame_id,
                    stamp_sec=decoded.metadata.stamp_sec,stamp_nanosec=decoded.metadata.stamp_nanosec,payload_equal=True,
                    message_owner_retained=decoded.owner is msg,view_readonly=not view.flags.writeable,
                    tensor_max_abs_error=float(np.abs(actual.numpy()-expected).max()),logit_max_abs_error=float((head-baseline).abs().max()),
                    top5=actual_labels,cat_probability=mass))
                self.count+=1
                if self.count==8:self.finish()
        node=Gate();rclpy.spin(node)
        if report['status']!='passed':raise RuntimeError('ROS stopped before acceptance')
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
