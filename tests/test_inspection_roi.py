"""Check ROI execution against independent full-frame numerical semantics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inspection_reference import numpy_inspection
from inspection_roi import TorchInspectionROI
from cpg import InspectionPipeline


@pytest.mark.parametrize('shape', [(1,1), (2,5), (7,9), (19,23)])
def test_roi_edges_and_retained_output(shape):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(4)
    h, w = shape
    frame = np.random.default_rng(1300).integers(0,256,(h,w,3),dtype=np.uint8)
    prepared = TorchInspectionROI('cpu')
    resident = torch.from_numpy(frame)
    saved = []
    rois = {(0,0,w,h), (0,0,1,1), (w-1,h-1,1,1),
            (0,h-1,w,1), (w-1,0,1,h), (w//2,h//2,1,1)}
    for roi in sorted(rois):
        actual = prepared.from_bgr8(resident,roi)
        expected = numpy_inspection(frame,roi)
        np.testing.assert_allclose(actual.numpy(),expected,atol=2e-4,rtol=0)
        saved.append((actual,actual.clone()))
    for actual, original in saved:
        torch.testing.assert_close(actual,original,atol=0,rtol=0)
    np.testing.assert_array_equal(resident.numpy(),frame)


def test_public_plan_contract():
    from dataclasses import FrozenInstanceError
    pipeline = InspectionPipeline((567,180,2706,1800),strategy='roi')
    plan = pipeline.plan((2160,3840,3))
    assert plan.work_box == (564,177,2712,1806)
    assert plan.processed_pixels == 2712*1806
    assert InspectionPipeline(pipeline.roi).plan((2160,3840,3)).work_box == (0,0,3840,2160)
    with pytest.raises(FrozenInstanceError):
        plan.strategy = 'full'
    with pytest.raises(ValueError,match='inside'):
        pipeline.plan((20,30,3))
    with pytest.raises(ValueError,match='reference=True'):
        pipeline.prepare((2160,3840,3),device='cpu')


@pytest.mark.parametrize('strategy',['full','roi'])
def test_public_runtime_numerics_and_retention(strategy):
    torch = pytest.importorskip('torch')
    frame = np.random.default_rng(15).integers(0,256,(19,23,3),dtype=np.uint8)
    for roi in [(0,0,23,19),(22,18,1,1),(4,3,12,11)]:
        prepared = InspectionPipeline(roi,strategy).prepare(frame.shape,device='cpu',reference=True)
        resident = torch.from_numpy(frame)
        actual = prepared(resident)
        np.testing.assert_allclose(actual.numpy(),numpy_inspection(frame,roi),atol=2e-4,rtol=0)
        saved = actual.clone()
        prepared(torch.zeros_like(resident))
        torch.testing.assert_close(actual,saved,atol=0,rtol=0)
        with pytest.raises(ValueError,match='shape and device'):
            prepared(resident[:2])
        with pytest.raises(TypeError,match='uint8'):
            prepared(resident.float())
        with torch.autocast('cpu'), pytest.raises(ValueError,match='autocast'):
            prepared(resident)
        with pytest.raises(ValueError,match='requires CUDA'):
            prepared.submit(resident)
        with pytest.raises(ValueError,match='Input event'):
            prepared(resident,ready_event=object())
