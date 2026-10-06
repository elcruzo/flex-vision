"""Experimental ROI execution with unchanged full-frame filter semantics."""
from inspection_reference import TorchInspectionBaseline


class TorchInspectionROI(TorchInspectionBaseline):
    """Restrict work to the requested ROI plus a three-pixel filter halo.

    Gaussian radius two followed by sharpen radius one needs three neighbors.
    Clip the work rectangle to the original image so replication still occurs
    at global borders. Keep both filter stages and their arithmetic unchanged.
    """

    def from_bgr8(self, frame, roi):
        import torch
        if not isinstance(frame, torch.Tensor) or frame.dtype != torch.uint8:
            raise TypeError('Expected a resident uint8 PyTorch tensor')
        if frame.device != self.device or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected HWC BGR on the prepared device')
        if len(roi) != 4 or any(type(v) is not int for v in roi):
            raise ValueError('ROI must contain integer x, y, width, height')
        x, y, w, h = roi
        height, width = frame.shape[:2]
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
            raise ValueError('ROI must be nonempty and inside the input')
        left, top = max(0, x-3), max(0, y-3)
        right, bottom = min(width, x+w+3), min(height, y+h+3)
        return super().from_bgr8(frame[top:bottom, left:right],
                                 (x-left, y-top, w, h))
