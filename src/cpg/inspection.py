"""Experimental fixed inspection recipe with explicit execution planning.

This API is separate from the detector Pipeline grammar. It is not autotuning.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class InspectionPlan:
    input_shape: tuple
    roi: tuple
    work_box: tuple
    strategy: str
    reason: str

    @property
    def processed_pixels(self):
        return self.work_box[2] * self.work_box[3]


@dataclass(frozen=True)
class InspectionResult:
    """CUDA output plus completion event. Use wait() on the consumer stream."""
    tensor: object
    ready: object

    def wait(self, stream=None):
        """Enqueue a device wait and protect output storage on the consumer."""
        import torch
        stream = torch.cuda.current_stream(self.tensor.device) if stream is None else stream
        if stream.device != self.tensor.device:
            raise ValueError('Consumer stream must use the output device')
        stream.wait_event(self.ready)
        self.tensor.record_stream(stream)
        return self.tensor


@dataclass(frozen=True)
class InspectionPipeline:
    """Gaussian 5/1.2, sharpen, clamp, crop, 224-square resize, normalize.

    Input is BGR8 HWC. Output is RGB FP32 NCHW with ImageNet normalization.
    Selection is explicit until tuning can justify a portable automatic policy.
    """
    roi: tuple
    strategy: str = 'full'

    def __post_init__(self):
        if type(self.roi) is not tuple or len(self.roi) != 4:
            raise ValueError('ROI must be an immutable x, y, width, height tuple')
        if any(type(v) is not int for v in self.roi):
            raise ValueError('ROI coordinates must be integers')
        x,y,w,h = self.roi
        if min(x,y) < 0 or min(w,h) <= 0:
            raise ValueError('ROI must be nonempty with nonnegative coordinates')
        if self.strategy not in ('full','roi'):
            raise ValueError('strategy must be full or roi')

    def plan(self, shape):
        shape = tuple(shape)
        if len(shape) != 3 or shape[2] != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('Expected a positive HWC shape with three channels')
        h,w,_ = shape
        x,y,rw,rh = self.roi
        if x+rw > w or y+rh > h:
            raise ValueError('ROI must be inside the input')
        if self.strategy == 'full':
            box = (0,0,w,h)
            reason = 'Explicit full-frame strategy; no automatic performance policy'
        else:
            left,top = max(0,x-3),max(0,y-3)
            right,bottom = min(w,x+rw+3),min(h,y+rh+3)
            box = (left,top,right-left,bottom-top)
            reason = 'Explicit ROI strategy with a three-pixel dependency halo'
        return InspectionPlan(shape,self.roi,box,self.strategy,reason)

    def prepare(self, shape, *, device='cuda', reference=False):
        """Allocate constants once. CPU/MPS require explicit reference=True."""
        return _PreparedInspection(self.plan(shape),device,reference)


class _PreparedInspection:
    """Enqueue on the current PyTorch stream and return a fresh owned tensor.

    Supply ready_event for another producer stream. Use submit for a consumer handoff.
    The caller must not overwrite input storage before preprocessing completes.
    CUDA execution does not synchronize or copy pixels to the host.
    """
    def __init__(self, plan, device, reference):
        import numpy as np
        import torch
        self.plan = plan
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not reference:
            raise ValueError('CPU/MPS require explicit reference=True')
        if self.device.type not in ('cpu','mps','cuda'):
            raise ValueError('Expected cpu, mps, or cuda device')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda',torch.cuda.current_device())
        x = np.arange(-2,3,dtype=np.float64)
        coeff = np.exp(-x*x/(2*1.2**2))
        coeff = torch.from_numpy((coeff/coeff.sum()).astype(np.float32)).to(self.device)
        self.device = coeff.device
        self.horizontal = coeff.view(1,1,1,5).repeat(3,1,1,1)
        self.vertical = coeff.view(1,1,5,1).repeat(3,1,1,1)
        self.kernel = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=torch.float32,device=self.device).view(1,1,3,3).repeat(3,1,1,1)
        self.mean = torch.tensor([.485,.456,.406],dtype=torch.float32,device=self.device).view(1,3,1,1)
        self.std = torch.tensor([.229,.224,.225],dtype=torch.float32,device=self.device).view(1,3,1,1)
        self._constants_ready = None
        self._creation_stream = None
        if self.device.type == 'cuda':
            self._creation_stream = torch.cuda.current_stream(self.device)
            self._constants_ready = torch.cuda.Event()
            self._constants_ready.record(self._creation_stream)

    def submit(self, frame, *, ready_event=None):
        """Return a completion handle for a CUDA consumer on another stream."""
        import torch
        if self.device.type != 'cuda':
            raise ValueError('submit requires CUDA execution')
        tensor = self(frame,ready_event=ready_event)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        return InspectionResult(tensor,ready)

    def __call__(self, frame, *, ready_event=None):
        import torch
        import torch.nn.functional as F
        if not isinstance(frame,torch.Tensor) or frame.dtype != torch.uint8:
            raise TypeError('Expected a resident uint8 PyTorch tensor')
        if frame.device != self.device or tuple(frame.shape) != self.plan.input_shape:
            raise ValueError('Input must match the prepared shape and device')
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError('Disable autocast for the FP32 inspection recipe')
        if self.device.type == 'cuda' and torch.backends.cudnn.allow_tf32:
            raise ValueError('Disable cuDNN TF32 for the FP32 inspection recipe')
        if ready_event is not None:
            if self.device.type != 'cuda' or not isinstance(ready_event,torch.cuda.Event) or ready_event.device != self.device:
                raise ValueError('Input event must be recorded on the prepared CUDA device')
        if self.device.type == 'cuda':
            stream = torch.cuda.current_stream(self.device)
            if stream != self._creation_stream:
                stream.wait_event(self._constants_ready)
                for tensor in (self.horizontal,self.vertical,self.kernel,self.mean,self.std):
                    tensor.record_stream(stream)
            if ready_event is not None:
                stream.wait_event(ready_event)
            # Waiting orders work. record_stream separately protects allocator lifetimes.
            frame.record_stream(stream)
        left,top,width,height = self.plan.work_box
        x,y,w,h = self.plan.roi
        x,y = x-left,y-top
        image = frame[top:top+height,left:left+width].flip(-1).permute(2,0,1).unsqueeze(0).to(torch.float32)
        image = F.conv2d(F.pad(image,(2,2,0,0),mode='replicate'),self.horizontal,groups=3)
        image = F.conv2d(F.pad(image,(0,0,2,2),mode='replicate'),self.vertical,groups=3)
        image = F.conv2d(F.pad(image,(1,1,1,1),mode='replicate'),self.kernel,groups=3).clamp(0,255)
        image = F.interpolate(image[:,:,y:y+h,x:x+w],size=(224,224),mode='bilinear',align_corners=False,antialias=False)
        return ((image*(1/255)-self.mean)/self.std).contiguous()
