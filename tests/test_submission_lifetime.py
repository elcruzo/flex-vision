"""Host lifecycle fault checks. These do not establish CUDA acceptance."""
import gc
import sys
from types import SimpleNamespace
import weakref
import pytest
from cpg.cuda import Submission


class Owner:
    pass


class Device:
    def __init__(self, device): self.device = device
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Stream:
    device_id = 0
    def __init__(self): self.waits = []
    def wait_event(self, event): self.waits.append(event)


class Event:
    def __init__(self): self.calls = 0; self.fail = False
    def synchronize(self):
        self.calls += 1
        if self.fail: raise RuntimeError('completion failed')


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(Device=Device, Stream=Stream)))


def test_handoff_does_not_release_source_before_completion():
    owner = Owner(); ref = weakref.ref(owner); output = Owner(); event = Event()
    submission = Submission(owner, owner, output, Stream(), event, 0)
    del owner; gc.collect()
    consumer = Stream()
    assert submission.wait_on(consumer) is output
    assert consumer.waits == [event]
    assert ref() is not None
    assert event.calls == 0
    assert submission.wait() is output
    gc.collect()
    assert ref() is None
    submission.close()
    assert event.calls == 1


def test_failed_completion_keeps_owners_and_can_retry():
    owner = Owner(); ref = weakref.ref(owner); event = Event(); event.fail = True
    submission = Submission(owner, owner, Owner(), Stream(), event, 0)
    del owner
    with pytest.raises(RuntimeError): submission.close()
    assert ref() is not None
    assert not submission._completed
    event.fail = False
    submission.close()
    assert ref() is None


def test_wrong_device_handoff_rejected_without_wait():
    event = Event(); submission = Submission(Owner(), Owner(), Owner(), Stream(), event, 0)
    consumer = Stream(); consumer.device_id = 1
    with pytest.raises(ValueError): submission.wait_on(consumer)
    assert not consumer.waits
    submission.close()
