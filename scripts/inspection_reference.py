"""Fixed inspection workload references. These are not CPG production operators."""
import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SHARPEN = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)


def gaussian_coefficients():
    x = np.arange(-2,3,dtype=np.float64)
    values = np.exp(-(x*x)/(2*1.2**2))
    return (values/values.sum()).astype(np.float32)


def validate(frame, roi):
    if not isinstance(frame,np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise TypeError('Expected a NumPy uint8 HWC BGR image')
    if len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError('ROI must contain integer x, y, width, height')
    x,y,w,h = roi
    if min(x,y)<0 or min(w,h)<=0 or x+w>frame.shape[1] or y+h>frame.shape[0]:
        raise ValueError('ROI must be nonempty and inside the input')


def numpy_resize(image, height=224, width=224):
    h,w,_ = image.shape
    y = np.maximum((np.arange(height,dtype=np.float64)+.5)*h/height-.5,0)
    x = np.maximum((np.arange(width,dtype=np.float64)+.5)*w/width-.5,0)
    y0,x0 = y.astype(np.int64),x.astype(np.int64)
    y1,x1 = np.minimum(y0+1,h-1),np.minimum(x0+1,w-1)
    wy,wx = (y-y0).astype(np.float32)[:,None,None],(x-x0).astype(np.float32)[None,:,None]
    a = image[y0[:,None],x0[None,:]]*(1-wx)+image[y0[:,None],x1[None,:]]*wx
    b = image[y1[:,None],x0[None,:]]*(1-wx)+image[y1[:,None],x1[None,:]]*wx
    return a*(1-wy)+b*wy


def numpy_inspection(frame, roi):
    validate(frame,roi)
    image = frame[...,::-1].astype(np.float32)
    coeff = gaussian_coefficients()
    padded = np.pad(image,((0,0),(2,2),(0,0)),mode='edge')
    horizontal = np.zeros_like(image)
    for k,c in enumerate(coeff):
        horizontal += padded[:,k:k+image.shape[1]]*c
    padded = np.pad(horizontal,((2,2),(0,0),(0,0)),mode='edge')
    blurred = np.zeros_like(image)
    for k,c in enumerate(coeff):
        blurred += padded[k:k+image.shape[0]]*c
    padded = np.pad(blurred,((1,1),(1,1),(0,0)),mode='edge')
    sharpened = np.zeros_like(image)
    for y in range(3):
        for x in range(3):
            sharpened += padded[y:y+image.shape[0],x:x+image.shape[1]]*SHARPEN[y,x]
    sharpened = np.clip(sharpened,0,255)
    x,y,w,h = roi
    resized = numpy_resize(sharpened[y:y+h,x:x+w])
    normalized = (resized*np.float32(1/255)-np.asarray(MEAN,dtype=np.float32))/np.asarray(STD,dtype=np.float32)
    return np.ascontiguousarray(normalized.transpose(2,0,1)[None])


def torch_inspection(frame, roi, *, device='cpu'):
    import torch
    import torch.nn.functional as F
    validate(frame,roi)
    if device not in ('cpu','mps'):
        raise ValueError('Local reference device must be cpu or mps')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    # This explicit upload is only for local reference validation.
    image = torch.from_numpy(np.array(frame[...,::-1],copy=True)).to(device=device,dtype=torch.float32)
    image = image.permute(2,0,1)[None]
    coeff = torch.from_numpy(gaussian_coefficients()).to(device)
    horizontal = coeff.view(1,1,1,5).repeat(3,1,1,1)
    vertical = coeff.view(1,1,5,1).repeat(3,1,1,1)
    image = F.conv2d(F.pad(image,(2,2,0,0),mode='replicate'),horizontal,groups=3)
    image = F.conv2d(F.pad(image,(0,0,2,2),mode='replicate'),vertical,groups=3)
    kernel = torch.from_numpy(SHARPEN.copy()).to(device).view(1,1,3,3).repeat(3,1,1,1)
    image = F.conv2d(F.pad(image,(1,1,1,1),mode='replicate'),kernel,groups=3).clamp(0,255)
    x,y,w,h = roi
    image = F.interpolate(image[:,:,y:y+h,x:x+w],size=(224,224),mode='bilinear',align_corners=False,antialias=False)
    mean = torch.tensor(MEAN,device=device).view(1,3,1,1)
    std = torch.tensor(STD,device=device).view(1,3,1,1)
    return ((image*(1/255)-mean)/std).contiguous()
