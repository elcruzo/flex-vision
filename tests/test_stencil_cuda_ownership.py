"""Dispatch/lifetime fault checks. No NVIDIA execution claim."""
import gc
import sys
from types import SimpleNamespace
import weakref
import numpy as np
import pytest
from cpg import Pipeline
from cpg import cuda
from cpg.stencil_cuda import source


class Array:
    def __init__(self,shape,dtype):
        self.shape=shape;self.dtype=np.dtype(dtype);self.size=__import__('math').prod(shape)
        self.strides=(shape[1]*3*self.dtype.itemsize,3*self.dtype.itemsize,self.dtype.itemsize)
        self.device=SimpleNamespace(id=0)
    def __dlpack_device__(self):return (2,0)


class Event:
    def __init__(self,**kwargs):self.finished=False
    def record(self,stream):pass
    def synchronize(self):self.finished=True


class Device:
    def __init__(self,n):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass


def test_workspace_retained_and_error_drain(monkeypatch):
    refs=[];calls=[]
    def allocate(shape,dtype):
        array=Array(shape,dtype);refs.append(weakref.ref(array));return array
    stream=SimpleNamespace(synchronize=lambda:calls.append('drain'))
    cp=SimpleNamespace(ndarray=Array,uint8=np.dtype('uint8'),float32=np.dtype('float32'),empty=allocate,
        cuda=SimpleNamespace(Event=Event,Device=Device,runtime=SimpleNamespace(getDevice=lambda:0),get_current_stream=lambda:stream))
    monkeypatch.setitem(sys.modules,'cupy',cp)
    import cpg.stencil_cuda as stencil
    monkeypatch.setattr(stencil,'kernel',lambda k:lambda *a,**kw:calls.append('stencil'))
    monkeypatch.setattr(cuda,'_kernel',lambda *a:lambda *args,**kw:calls.append('preprocess'))
    p=Pipeline().conv2d([[0,0,0],[0,1,0],[0,0,0]]).letterbox(2,2).normalize([0]*3,[1]*3,scale=1).to(dtype='float16',layout='NCHW')
    pending=cuda._execute(p,Array((5,7,3),'uint8'),out=None,asynchronous=True)
    gc.collect()
    assert calls==['stencil','preprocess'] and refs[1]() is not None
    pending.close();gc.collect()
    assert refs[1]() is None
    def fail(*args):
        assert refs[-1]() is not None
        raise RuntimeError('compile failure after stencil dispatch')
    monkeypatch.setattr(cuda,'_kernel',fail)
    with pytest.raises(RuntimeError,match='compile failure'):cuda.execute(p,Array((5,7,3),'uint8'))
    assert calls[-2:]==['stencil','drain']


@pytest.mark.parametrize('size',[3,5,7,9])
def test_specialization_has_no_runtime_coefficient_loop(size):
    coefficients=tuple(tuple(.125 if x==y==0 else 0. for x in range(size)) for y in range(size))
    generated=source(coefficients)
    assert 'for (' not in generated and 'while (' not in generated
    assert '1.250000000e-01f' in generated
    assert generated.count('value = value +')==size*size
