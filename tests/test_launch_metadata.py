"""Host plan cache must respect immutable graph and input contracts."""
import numpy as np
import pytest
from cpg import Pipeline
from cpg.cuda import _launch_metadata


def test_metadata_cache_separates_graph_shape_and_precision():
    _launch_metadata.cache_clear()
    pipe = Pipeline(input_encoding='bgr8').letterbox(640,640).normalize([0]*3,[1]*3,scale=1/255).to(dtype='float16',layout='NCHW')
    first = _launch_metadata(pipe,(1080,1920,3))
    assert _launch_metadata(pipe,(1080,1920,3)) is first
    assert first[:3] == ((1,3,640,640),'float16',3*640*640)
    odd = _launch_metadata(pipe,(301,451,3))
    assert tuple(first[3][:8]) != tuple(odd[3][:8])
    changed = Pipeline().letterbox(320,320).normalize([.1]*3,[.2]*3,scale=1).to(dtype='float32',layout='NCHW')
    other = _launch_metadata(changed,(1080,1920,3))
    assert other[:3] == ((1,3,320,320),'float32',3*320*320)
    assert other[3][8] == 0 and first[3][8] == 1
    assert all(isinstance(value,np.generic) for value in first[3])
    assert _launch_metadata.cache_info().maxsize == 64
    with pytest.raises(ValueError,match='three channels'):
        _launch_metadata(pipe,(1080,1920,4))
