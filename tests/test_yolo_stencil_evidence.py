"""Reject incomplete stencil inference reports without GPU claims."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from verify_yolo_stencil import verify,FIXTURES


def evidence():
    checks=[]
    for f in sorted(FIXTURES):
        for s in (3,5,7,9):
            for v in ('positive','negative'):
                for l in ('contiguous','reverse_channels'):
                    for e in ('sync','submit'):
                        checks.append(dict(fixture=f,size=s,variant=v,layout=l,execution=e,tensor_exact=True,dense_output=True,
                            detections=True,retained_output_exact=True if checks else None,
                            plan=dict(cuda_launches=2,temporary_arrays=1,stencil_implementation='specialized-direct-experimental')))
    return dict(status='passed',drain_status='passed',dirty=False,mode='stencil-inference-v1',revision='a'*40,engine_sha256='b'*64,checks=checks)


def test_complete():assert verify(evidence())['executions']==192


@pytest.mark.parametrize('fault',['missing','duplicate','tensor','plan','dirty'])
def test_incomplete_rejected(fault):
    r=evidence()
    if fault=='missing':r['checks'].pop()
    elif fault=='duplicate':r['checks'][-1]=r['checks'][0]
    elif fault=='tensor':r['checks'][0]['tensor_exact']=False
    elif fault=='plan':r['checks'][0]['plan']['temporary_arrays']=0
    else:r['dirty']=True
    with pytest.raises(ValueError):verify(r)


def gaussian_evidence():
    from cpg import Pipeline
    r=evidence();r.update(mode='stencil-inference-v2',gaussian_sigma=1.2)
    for f in sorted(FIXTURES):
        for size in (3,5,7,9):
            kernel=Pipeline().gaussian(size,sigma=1.2).operations[0].kernel
            for layout in ('contiguous','reverse_channels'):
                for execution in ('sync','submit'):
                    r['checks'].append(dict(fixture=f,size=size,variant='gaussian',layout=layout,execution=execution,
                        tensor_exact=True,dense_output=True,detections=True,retained_output_exact=True,
                        plan=dict(cuda_launches=2,temporary_arrays=1,stencil_implementation='specialized-direct-experimental',
                                  operations=[dict(kind='Conv2d',kernel=kernel)])))
    return r


def test_complete_gaussian_matrix():
    assert verify(gaussian_evidence())['executions']==288


@pytest.mark.parametrize('fault',['no-gaussian','wrong-kernel','wrong-sigma'])
def test_gaussian_claim_requires_dense_coefficients_and_full_coverage(fault):
    r=gaussian_evidence()
    if fault=='no-gaussian':r['checks']=r['checks'][:192]
    elif fault=='wrong-kernel':r['checks'][-1]['plan']['operations'][0]['kernel']=((1.,),)
    else:r['gaussian_sigma']=2
    with pytest.raises(ValueError):verify(r)
