"""Experimental configured ROS preprocessing node. CPU reference is explicit."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu-reference',), required=True)
    parser.add_argument('--input-topic', default='cpg_reference/image')
    parser.add_argument('--output-topic', default='cpg_reference/tensor')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--queue-depth', type=int, default=2)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error('CPU thread count must be positive')
    if args.queue_depth <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error('Queue depth and frame limit must be positive')
    if args.deadline is not None and (not 0 < args.deadline < float('inf')):
        parser.error('Deadline must be finite and positive')
    if args.report is not None and args.report.exists():
        parser.error('Report already exists')
    import torch
    torch.set_num_threads(args.cpu_threads)
    from .config import load_pipeline
    from .reference import torch_reference
    from .ros import cpu_image_view, write_cpu_tensor, write_tensor_geometry
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from cpg_tensor_probe.msg import StampedTensor
    pipeline = load_pipeline(args.config)
    report = dict(status='failed', backend=args.backend, frames=0, checks=[],
                  scope='Experimental CPU reference node; host payload copies; no native CUDA bridge')
    rclpy.init()
    node = None
    try:
        class PreprocessNode(Node):
            def __init__(self):
                super().__init__('cuda_preprocess_node')
                self.started = time.monotonic()
                self.finished_at = None
                self.qos = QoSProfile(depth=args.queue_depth, reliability=ReliabilityPolicy.RELIABLE)
                self.publisher = self.create_publisher(StampedTensor, args.output_topic, self.qos)
                self.subscription = None
                self.timer = self.create_timer(.1, self.tick)
            def tick(self):
                now = time.monotonic()
                if args.deadline is not None and now-self.started > args.deadline:
                    raise TimeoutError('Preprocessing node deadline exceeded')
                if self.finished_at is not None:
                    if now-self.finished_at >= 1:
                        report['status'] = 'passed'
                        rclpy.shutdown()
                    return
                if self.subscription is None and self.publisher.get_subscription_count():
                    self.subscription = self.create_subscription(Image, args.input_topic, self.receive, self.qos)
            def receive(self, message):
                if self.finished_at is not None:
                    raise RuntimeError('Received input beyond the declared frame limit')
                if not self.publisher.get_subscription_count():
                    raise RuntimeError('Tensor consumer disappeared')
                decoded = cpu_image_view(message, expected_encoding=pipeline.input_encoding)
                output, geometry = torch_reference(pipeline, decoded.pixels, device='cpu')
                result = StampedTensor()
                result.header = message.header
                result.layout = 'NCHW'
                result.source_width = decoded.metadata.width
                result.source_height = decoded.metadata.height
                result.source_encoding = decoded.metadata.encoding
                write_cpu_tensor(result.tensor, output.numpy())
                write_tensor_geometry(result, geometry)
                self.publisher.publish(result)
                report['checks'].append(dict(sequence=report['frames'], frame_id=result.header.frame_id,
                    stamp_sec=result.header.stamp.sec, stamp_nanosec=result.header.stamp.nanosec,
                    shape=list(output.shape), payload_bytes=len(result.tensor.data)))
                report['frames'] += 1
                if args.max_frames is not None and report['frames'] == args.max_frames:
                    self.finished_at = time.monotonic()
        node = PreprocessNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
