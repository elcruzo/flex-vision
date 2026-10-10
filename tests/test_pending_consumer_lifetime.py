"""Host fault checks only, not CUDA/TensorRT acceptance."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import weakref
import pytest


@pytest.fixture
def classes(monkeypatch):
    class Device:
        def __init__(self, device): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(Device=Device)))
    monkeypatch.setitem(sys.modules, 'yolo_gpu', SimpleNamespace(Consumer=object))
    spec = importlib.util.spec_from_file_location('pending_consumer_test', Path(__file__).parents[1] / 'scripts/pending_yolo_consumer.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Owner: pass


class Event:
    fail = False
    calls = 0
    def synchronize(self):
        self.calls += 1
        if self.fail: raise RuntimeError('completion failed')


def test_consumer_retains_input_and_context_until_completion(classes):
    consumer = SimpleNamespace(device=0, _busy=True)
    image = Owner(); ref = weakref.ref(image); event = Event(); output = Owner()
    job = classes.PendingInference(consumer, image, output, event)
    del image
    assert ref() is not None
    assert consumer._busy
    assert job.wait() is output
    assert ref() is None
    assert not consumer._busy
    job.close()
    assert event.calls == 1


def test_failed_completion_preserves_owners_and_busy_context(classes):
    consumer = SimpleNamespace(device=0, _busy=True)
    image = Owner(); ref = weakref.ref(image); event = Event(); event.fail = True
    job = classes.PendingInference(consumer, image, Owner(), event)
    del image
    with pytest.raises(RuntimeError): job.close()
    assert ref() is not None and consumer._busy
    event.fail = False; job.close()
    assert ref() is None and not consumer._busy


def test_busy_context_rejects_rebinding_before_dispatch(classes):
    consumer = classes.PendingConsumer.__new__(classes.PendingConsumer)
    consumer._busy = True
    with pytest.raises(RuntimeError): consumer.enqueue(Owner())
