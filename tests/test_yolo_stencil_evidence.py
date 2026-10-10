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
