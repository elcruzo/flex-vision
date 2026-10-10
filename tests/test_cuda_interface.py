"""Metadata rejection and borrowed-owner handoff. No CUDA acceptance claim."""
from types import SimpleNamespace
import pytest
from cpg.interop import validate_cuda_interface, import_cuda_interface


def metadata():
    return dict(version=3, shape=(5, 7, 3), typestr='|u1', data=(1234, False), strides=(21, 3, 1), stream=8)


@pytest.mark.parametrize('field,value', [('version',2), ('stream',0), ('stream',True), ('stream',-1),
    ('shape',(0,7,3)), ('shape',(5,7,4)), ('data',(0,False)), ('mask',object()),
    ('strides',(21,3)), ('typestr','<f4')])
def test_invalid_metadata_rejected(field, value):
    item=metadata();item[field]=value
    with pytest.raises((ValueError,TypeError)):validate_cuda_interface(item)


def test_snapshot_retains_signed_strides_and_readonly():
    item=metadata();item['strides']=(-21,3,-1);item['data']=(1234,True)
    result=validate_cuda_interface(item);item['shape']=(1,1,3)
    assert result['shape']==(5,7,3) and result['strides']==(-21,3,-1)
    assert result['data'][1] is True


class Stream:
    def __init__(self):self.waits=[]
    def wait_event(self,event):self.waits.append(event)


class Event:
    def __init__(self,**kwargs):self.recorded=None
    def record(self,stream):self.recorded=stream


@pytest.mark.parametrize('pointer',[None,1,2,8])
def test_import_orders_producer_and_retains_owner(pointer):
    frame=SimpleNamespace(__cuda_array_interface__=metadata());frame.__cuda_array_interface__['stream']=pointer
    current=Stream();legacy=Stream();ptds=Stream();external=Stream()
    external.owner=None
    def from_external(owner):
        assert owner.__cuda_stream__()==(0,8)
        external.owner=owner
        return external
    def asarray(view):
        # The adapter handles ordering itself, independent of CuPy's sync setting.
        assert view.__cuda_array_interface__['stream'] is None
        assert view.owner is frame
        return SimpleNamespace(device=SimpleNamespace(id=0), data=SimpleNamespace(ptr=1234),shape=(5,7,3),dtype='uint8',strides=(21,3,1))
    cp=SimpleNamespace(asarray=asarray,uint8='uint8',cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda:0),get_current_stream=lambda:current,Event=Event,
        Stream=SimpleNamespace(null=legacy,ptds=ptds,from_external=from_external)))
    source,owners,producer=import_cuda_interface(frame,cp)
    assert owners[0] is frame and source.data.ptr==1234
    if pointer is None:
        assert producer is None and current.waits==[]
    else:
        assert producer is {1:legacy,2:ptds,8:external}[pointer]
        assert current.waits[0].recorded is producer
        assert current.waits[0] in owners
