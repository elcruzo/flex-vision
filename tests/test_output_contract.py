"""Output metadata checks prevent wrong writes before any CUDA dispatch."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from cpg.cuda import _validate_output,_byte_interval


def array(shape,dtype,pointer,strides,device=0,contiguous=True):
    return NS(shape=shape,dtype=np.dtype(dtype),data=NS(ptr=pointer),strides=strides,
              device=NS(id=device),flags=NS(c_contiguous=contiguous))


def test_output_contract_and_negative_input_extent():
    source=array((2,2,3),'uint8',1012,(-6,3,1))
    assert _byte_interval(source)==(1006,1018)
    out=array((1,3,2,2),'float16',2000,(24,8,4,2))
    _validate_output(out,source,(1,3,2,2),'float16',0)
    out.data.ptr=1016
    with pytest.raises(ValueError,match='overlap'):_validate_output(out,source,(1,3,2,2),'float16',0)


@pytest.mark.parametrize('field,value,error',[
    ('device',NS(id=1),'device'),('shape',(1,3,2,3),'shape'),
    ('dtype',np.dtype('float32'),'dtype'),('flags',NS(c_contiguous=False),'contiguous'),
    ('data',NS(ptr=2001),'aligned')])
def test_output_metadata_rejects_invalid_destinations(field,value,error):
    source=array((2,2,3),'uint8',1000,(6,3,1))
    out=array((1,3,2,2),'float16',2000,(24,8,4,2))
    setattr(out,field,value)
    with pytest.raises((ValueError,TypeError),match=error):
        _validate_output(out,source,(1,3,2,2),'float16',0)
