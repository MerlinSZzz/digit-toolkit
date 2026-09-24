"""Sensor-free tests for optical flow and slip detection."""

import cv2
import numpy as np

from digit import flow as F


def _textured(shape=(120, 160)):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, shape, dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def test_dense_flow_recovers_translation():
    prev = _textured()
    dx, dy = 3.0, 2.0
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    curr = cv2.warpAffine(prev, matrix, (prev.shape[1], prev.shape[0]))
    flow = F.dense_flow(prev, curr)
    # look away from the translated border
    interior = flow[20:-20, 20:-20]
    assert abs(np.median(interior[..., 0]) - dx) < 1.0
    assert abs(np.median(interior[..., 1]) - dy) < 1.0


def test_flow_vis_shape_and_range():
    prev = _textured()
    flow = F.dense_flow(prev, prev)
    vis, mag = F.flow_vis(flow)
    assert vis.shape == (120, 160, 3)
    assert vis.dtype == np.uint8
    assert mag.shape == (120, 160)


def test_sparse_flow_returns_points():
    prev = _textured()
    matrix = np.float32([[1, 0, 2], [0, 1, 1]])
    curr = cv2.warpAffine(prev, matrix, (prev.shape[1], prev.shape[0]))
    p0, p1, status = F.sparse_flow(prev, curr)
    assert len(p0) > 0
    assert status.sum() > 0


def test_shear_from_sparse_background_subtraction():
    # 3 background corners moving 5 px, 3 contact corners moving 1 px
    p0 = np.array(
        [[[10.0, 10.0]], [[20.0, 20.0]], [[30.0, 30.0]], [[50.0, 50.0]], [[55.0, 50.0]], [[60.0, 50.0]]],
        np.float32,
    )
    disp = np.array(
        [[[5.0, 0.0]], [[5.0, 0.0]], [[5.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]],
        np.float32,
    )
    p1 = p0 + disp
    status = np.ones((6, 1), np.uint8)
    mask = np.zeros((120, 160), np.uint8)
    mask[45:70, 45:70] = 255  # only the last three points are in contact
    result = F.shear_from_sparse(p0, p1, status, mask)
    # inside 1.0, outside 5.0 -> shear ≈ -4.0
    assert abs(result["dx"] - (-4.0)) < 0.5


def test_slip_detector_triggers_on_sustained_shear():
    detector = F.SlipDetector(shear_threshold=0.5, required=3)
    assert not detector.update(None, 1.0).slip
    assert not detector.update(None, 1.0).slip
    assert detector.update(None, 1.0).slip


def test_slip_detector_centroid_speed():
    detector = F.SlipDetector(shear_threshold=10.0, centroid_threshold=1.0, required=2)
    detector.update((0.0, 0.0), 0.0)
    assert not detector.update((5.0, 0.0), 0.0).slip
    assert detector.update((12.0, 0.0), 0.0).slip
