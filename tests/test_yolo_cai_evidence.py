"""Coverage fault checks, not NVIDIA acceptance."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from verify_yolo_cai import verify, FIXTURES


def evidence():
    rows=[]
    for f in sorted(FIXTURES):
        for l in ('contiguous','reverse_rows','reverse_channels'):
            for p in ('explicit','legacy','ptds','ready'):
                for e in ('sync','submit'):
                    rows.append(dict(fixture=f,layout=l,producer=p,execution=e,pending_producer=p!='ready',
                        reuse_ordering='caller_completion' if p=='ready' else 'advertised_stream_fence',
                        pending_at_reuse=True if e=='submit' and p!='ready' else None,
                        preprocessing_pending=True if e=='submit' else None,exporter_retained=True if e=='submit' else None,retained_output_exact=True if rows else None,
                        producer_reuse=True,tensor_exact=True,dense_output=True,detections=True))
    return dict(status='passed',dirty=False,drain_status='passed',mode='cai-v2',delay_cycles=500000000,preprocessing_delay_cycles=1000000000,
                revision='a'*40,engine_sha256='b'*64,checks=rows)


def test_complete_matrix():assert verify(evidence())['executions']==144


@pytest.mark.parametrize('fault',['missing','duplicate','producer','lifetime','retention','inference','dirty','mode'])
def test_incomplete_matrix_rejected(fault):
    report=evidence()
    if fault=='missing':report['checks'].pop()
    elif fault=='duplicate':report['checks'][-1]=report['checks'][0]
    elif fault=='producer':report['checks'][1]['pending_producer']=False
    elif fault=='lifetime':report['checks'][1]['exporter_retained']=False
    elif fault=='retention':report['checks'][1]['retained_output_exact']=False
    elif fault=='inference':report['checks'][1]['dense_output']=False
    elif fault=='dirty':report['dirty']=True
    else:report['mode']='async-submit'
    with pytest.raises(ValueError):verify(report)


def test_cupy_pending_check_uses_documented_done_property():
    from types import SimpleNamespace
    from yolo_cai import require_pending
    require_pending(SimpleNamespace(done=False), 'Producer')
    with pytest.raises(AssertionError, match='completed before'):
        require_pending(SimpleNamespace(done=True), 'Preprocessing')


@pytest.mark.parametrize('producer,execution', [('ready','submit'),('explicit','submit'),('legacy','submit'),('ptds','submit')])
def test_reuse_boundary_required(producer,execution):
    report=evidence()
    row=next(r for r in report['checks'] if r['producer']==producer and r['execution']==execution)
    row['reuse_ordering']='unordered'
    with pytest.raises(ValueError):verify(report)
    report=evidence()
    row=next(r for r in report['checks'] if r['producer']==producer and r['execution']==execution)
    row['pending_at_reuse']=False
    with pytest.raises(ValueError):verify(report)
