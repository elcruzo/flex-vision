"""Check ROI execution against independent full-frame numerical semantics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inspection_reference import numpy_inspection
from inspection_roi import TorchInspectionROI


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
