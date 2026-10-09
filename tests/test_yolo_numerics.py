"""Fault checks for the revised numerical acceptance policy."""
from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from yolo_common import compare_dense


def dense():
    result=torch.zeros((1,84,8400))
    result[:,:4]=10
    return result


def test_background_boxes_are_reported_and_scores_still_checked():
    a,b=dense(),dense()
    b[0,2,0]=100
    result=compare_dense(a,b,preprocessing=True)
    assert result['background_box_limit_violations'] == 1
    assert result['box_max_abs_error'] == 90
    with pytest.raises(AssertionError): compare_dense(a,b)
    b[0,4,1]=.005  # Below the box floor, but above the tightened score tolerance.
    with pytest.raises(AssertionError): compare_dense(a,b,preprocessing=True)


def test_box_floor_uses_both_reference_and_candidate_scores():
    a,b=dense(),dense()
    b[0,2,0]=100
    for selected in (a,b):
        selected[0,4,0]=.02
        with pytest.raises(AssertionError): compare_dense(a,b,preprocessing=True)
        selected[0,4,0]=0


@pytest.mark.parametrize('index,value', [((0,2,0),-1),((0,4,0),1.1),((0,0,0),float('nan'))])
def test_invalid_outputs_never_pass(index,value):
    a,b=dense(),dense();b[index]=value
    with pytest.raises(AssertionError): compare_dense(a,b,preprocessing=True)
