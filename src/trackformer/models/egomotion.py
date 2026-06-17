"""
Plan A — Egomotion-aware track query compensation.
Estimates dominant camera translation from consecutive egocentric frames
using dense optical flow, returning a normalised (dx, dy) offset.
"""
import cv2
import numpy as np
import torch

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _tensor_to_gray(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) normalised image tensor → uint8 grayscale numpy."""
    img = t.detach().cpu().float().numpy().transpose(1, 2, 0)  # H,W,C
    img = img * _STD + _MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def estimate_egomotion(prev_tensor: torch.Tensor,
                       curr_tensor: torch.Tensor) -> torch.Tensor:
    """
    Estimate camera egomotion between two consecutive egocentric frames.

    Args:
        prev_tensor: (C, H, W) normalised tensor — previous frame.
        curr_tensor: (C, H, W) normalised tensor — current frame.

    Returns:
        Float32 tensor [dx_norm, dy_norm] — camera translation in
        normalised image coordinates (divided by frame width / height).
        Positive dx = camera moved right; positive dy = camera moved down.
    """
    prev_gray = _tensor_to_gray(prev_tensor)
    curr_gray = _tensor_to_gray(curr_tensor)
    #____________ added for matching size with plan 1's size. otherwise optical flow can't be computed correctly
    # Resize prev to match curr if augmentation changed sizes
    if prev_gray.shape != curr_gray.shape:
        h, w = curr_gray.shape
        prev_gray = cv2.resize(prev_gray, (w, h))
    #__________________________________________________________
    h, w = prev_gray.shape

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )  # (H, W, 2)

    # Median suppresses contributions from moving objects — captures camera motion
    dx = float(np.median(flow[..., 0])) / w
    dy = float(np.median(flow[..., 1])) / h

    return torch.tensor([dx, dy], dtype=torch.float32)

