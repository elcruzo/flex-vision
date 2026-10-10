"""Stencil direction, border, configuration, and explicit support boundaries."""
import numpy as np
import pytest
from cpg import Pipeline, load_pipeline, ConfigError
from cpg.reference import numpy_reference, torch_reference


def pipe(kernel):
    return Pipeline().conv2d(kernel).letterbox(5,4).normalize([0]*3,[1]*3,scale=1).to(dtype='float32',layout='NCHW')


@pytest.mark.parametrize('size',[3,5,7,9])
def test_asymmetric_flipped_convolution_and_edge_replication(size):
    image=np.arange(60,dtype=np.uint8).reshape(4,5,3)
    kernel=np.zeros((size,size)).tolist();kernel[0][0]=1
    p=pipe(kernel)
    expected=image[np.minimum(np.arange(4)+size//2,3)[:,None],np.minimum(np.arange(5)+size//2,4)[None,:]]
    result,_=numpy_reference(p,image)
    np.testing.assert_array_equal(result[0].transpose(1,2,0),expected)
    actual,_=torch_reference(p,image)
    np.testing.assert_array_equal(actual.numpy(),result)
    plan=p.plan(image.shape,backend='cuda')
    assert (plan['cuda_launches'],plan['temporary_arrays'])==(2,1)


def test_negative_intermediate_is_not_clamped():
    kernel=[[0,0,0],[0,-1,0],[0,0,0]]
    result,_=numpy_reference(pipe(kernel),np.full((4,5,3),9,dtype=np.uint8))
    assert (result==-9).all()


def test_kernel_file_and_inline_graphs_match_and_snapshot(tmp_path):
    header='input: {encoding: rgb8}\npipeline:\n  - convolution: {kernel: kernel.yaml}\n  - letterbox: {width: 5, height: 4}\n  - normalize: {mean: [0,0,0], std: [1,1,1], scale: 1}\noutput: {dtype: float32, layout: nchw}\n'
    (tmp_path/'pipe.yaml').write_text(header)
    kernel=[[0,0,0],[0,1,0],[0,0,0]]
    (tmp_path/'kernel.yaml').write_text('kernel: '+str(kernel))
    p=load_pipeline(tmp_path/'pipe.yaml')
    assert p==pipe(kernel)
    (tmp_path/'kernel.yaml').write_text('kernel: [[1]]')
    assert p==pipe(kernel)
    with pytest.raises(ConfigError,match='convolution'):load_pipeline(tmp_path/'pipe.yaml')


@pytest.mark.parametrize('kernel', [[[1]],[[0]*3]*2,[[0]*3,[0,float('nan'),0],[0]*3],[[0]*3,[0,1e100,0],[0]*3]])
def test_invalid_stencils_rejected(kernel):
    with pytest.raises(ValueError):pipe(kernel)
