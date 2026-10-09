"""Admission checks cover late polling, bounded queues, and complete identities."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from yolo_load import Arrivals


def test_delayed_dispatch_preserves_all_arrivals_and_bounds_each_camera():
    arrivals=Arrivals(4,60,1,2,1)
    arrivals.advance(1.)
    assert len(arrivals.rows)==240
    assert arrivals.max_pending==8
    assert sum(row['status']=='dropped_queue_full' for row in arrivals.rows)==232
    admitted=[]
    while (row:=arrivals.pop()) is not None:admitted.append(row['frame'])
    assert admitted==list(range(8))
    assert arrivals.finished
    assert [row['lane'] for row in arrivals.rows[:4]]==[1,2,3,0]


def test_advance_does_not_admit_future_or_duplicate_frames():
    arrivals=Arrivals(4,1,1,2)
    arrivals.advance(0)
    assert arrivals.pop()['frame']==0
    arrivals.advance(.24)
    assert arrivals.pop() is None
    arrivals.advance(.25)
    assert arrivals.pop()['frame']==1
    arrivals.advance(1)
    assert [arrivals.pop()['frame'],arrivals.pop()['frame']]==[2,3]
    assert arrivals.finished
