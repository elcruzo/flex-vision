"""Experimental serial nonblocking TensorRT consumer for ownership validation."""
import cupy as cp
import torch
from yolo_gpu import Consumer


class PendingInference:
    def __init__(self, consumer, image, output, event):
        self._consumer = consumer
        self._image = image
        self._output = output
        self._event = event
        self._finished = False

    def wait(self):
        if not self._finished:
            with cp.cuda.Device(self._consumer.device):
                self._event.synchronize()
            self._finished = True
            self._image = None
            self._consumer._busy = False
        return self._output

    def close(self):
        self.wait()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def __del__(self):
        if hasattr(self, '_finished') and not self._finished:
            self.close()


class PendingConsumer(Consumer):
    """One outstanding enqueue. The native CuPy stream owner stays retained."""
    def __init__(self, engine, stream):
        super().__init__(engine)
        self.device = torch.cuda.current_device()
        if not isinstance(stream, cp.cuda.Stream) or stream.device_id != self.device:
            raise ValueError('Require an explicit current-device CuPy stream')
        self._native_stream = stream
        self._torch_stream = torch.cuda.ExternalStream(stream.ptr, device=self.device)
        self._busy = False

    def enqueue(self, image):
        if self._busy:
            raise RuntimeError('Complete the preceding inference before enqueue')
        if (not isinstance(image, cp.ndarray) or image.shape != (1, 3, 640, 640)
                or image.dtype != cp.float16 or not image.flags.c_contiguous
                or image.device.id != self.device or torch.cuda.current_device() != self.device):
            raise ValueError('Require current-device contiguous FP16 1x3x640x640 input')
        with torch.cuda.stream(self._torch_stream):
            result = torch.empty((1, 84, 8400), device='cuda', dtype=self.dtype)
            event = torch.cuda.Event()
            self._busy = True
            try:
                if not self.context.set_tensor_address('image', image.data.ptr):
                    raise RuntimeError('Input binding failed')
                if not self.context.set_tensor_address('predictions', result.data_ptr()):
                    raise RuntimeError('Output binding failed')
                if not self.context.execute_async_v3(self._torch_stream.cuda_stream):
                    raise RuntimeError('TensorRT enqueue failed')
                event.record(self._torch_stream)
                return PendingInference(self, image, result, event)
            except BaseException:
                self._torch_stream.synchronize()
                self._busy = False
                raise

    def __call__(self, image):
        # Preserve the serial guard even for a synchronous call on this context.
        return self.enqueue(image).wait()
