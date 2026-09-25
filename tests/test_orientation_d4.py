"""Sensor-free tests for the D4 orientation / deformation / texture-slip code."""

from __future__ import annotations

import json
import os
import tempfile

import cv2
import numpy as np

from digit import flow as F
from digit import orientation as O
from digit import processing as P


# ---------------------------------------------------------------------------
# orientation


def test_orient_flips_and_rotate():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert O.orient(a, flip_x=True)[0].tolist() == [3, 2, 1, 0]
    assert O.orient(a, flip_y=True)[:, 0].tolist() == [8, 4, 0]
    assert O.orient(a, rotate=180)[0].tolist() == [11, 10, 9, 8]
    assert O.orient(a, rotate=90).shape == (4, 3)
    assert O.orient(a, rotate=270).shape == (4, 3)


def test_operator_preset_is_flip_x_of_official():
    frame = np.arange(20, dtype=np.uint8).reshape(4, 5)
    official = O.apply_preset(frame, "official")
    operator = O.apply_preset(frame, "operator")
    assert np.array_equal(official, frame)
    assert np.array_equal(operator, cv2.flip(frame, 1))
    assert O.resolve("operator") == {"rotate": 0, "flip_x": True, "flip_y": False, "raw": False}
    assert O.resolve("official")["flip_x"] is False
    assert O.resolve("raw")["raw"] is True


def test_session_frame_orientation_defaults_to_official():
    with tempfile.TemporaryDirectory() as d:
        assert O.session_frame_orientation(d) == "official"
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"frame_orientation": "operator"}, fh)
        assert O.session_frame_orientation(d) == "operator"
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"width": 640}, fh)
        assert O.session_frame_orientation(d) == "official"


def test_to_operator_mirrors_official_only():
    frame = np.arange(20, dtype=np.uint8).reshape(4, 5)
    assert np.array_equal(O.to_operator(frame, "operator"), frame)
    assert np.array_equal(O.to_operator(frame, "official"), cv2.flip(frame, 1))
    assert np.array_equal(O.to_operator(frame, "raw"), cv2.flip(frame, 1))


# ---------------------------------------------------------------------------
# lighting-invariant deformation localisation


def _reference_with_texture(shape=(160, 160)):
    rng = np.random.default_rng(0)
    base = np.full(shape, 120, np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    base += 8.0 * np.sin(xx / 11.0) + 6.0 * np.cos(yy / 13.0)
    base += rng.normal(0.0, 0.6, shape).astype(np.float32)
    return cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_deformation_centroid_recovers_a_local_dipole():
    ref = _reference_with_texture()
    frame = ref.astype(np.float32)
    cx, cy = 105, 55
    # a local bright/dark dipole: an indentation lit from the left
    yy, xx = np.mgrid[0:ref.shape[0], 0:ref.shape[1]]
    g = np.exp(-(((xx - (cx + 8)) ** 2 + (yy - cy) ** 2) / (2 * 6.0 ** 2)))
    d = np.exp(-(((xx - (cx - 8)) ** 2 + (yy - cy) ** 2) / (2 * 6.0 ** 2)))
    frame = np.clip(frame + 30.0 * g[..., None] - 30.0 * d[..., None], 0, 255).astype(np.uint8)
    dmap = P.deformation_map(frame, ref, sigma=40.0, post_sigma=10.0)
    pred = P.deformation_centroid(dmap, quantile=0.3)
    assert pred is not None
    assert abs(pred[0] - cx) < 12 and abs(pred[1] - cy) < 12


def test_deformation_map_ignores_a_broad_brightness_shift():
    ref = _reference_with_texture()
    shifted = np.clip(ref.astype(np.float32) + 25.0, 0, 255).astype(np.uint8)
    dmap = P.deformation_map(shifted, ref, sigma=40.0, post_sigma=10.0)
    assert float(np.abs(dmap).max()) < 4.0  # a smooth lighting change is removed


# ---------------------------------------------------------------------------
# texture-motion slip


def test_frame_change_energy_separates_static_and_moving_texture():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    b = np.roll(a, 2, axis=1)
    mask = np.ones((64, 64), np.uint8) * 255
    static = F.frame_change_energy(a, a, mask=mask)
    moving = F.frame_change_energy(a, b, mask=mask)
    assert static == 0.0
    assert moving > 5.0


def test_texture_slip_detector_needs_touch_and_motion():
    rng = np.random.default_rng(2)
    frames = []
    base = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    for k in range(12):
        frames.append(np.roll(base, -k, axis=1))
    det = F.TextureSlipDetector(threshold=2.0, required=2, lag=3)
    mask = np.ones((64, 64), np.uint8) * 255
    # no touch -> never slip, even though the texture moves
    for f in frames:
        assert det.update(f, touch=False, mask=mask).slip is False
    # touch + motion -> slips
    det = F.TextureSlipDetector(threshold=2.0, required=2, lag=3)
    fired = False
    for f in frames:
        fired = fired or det.update(f, touch=True, mask=mask).slip
    assert fired
    # touch + still texture -> never slips
    det = F.TextureSlipDetector(threshold=2.0, required=2, lag=3)
    still = np.zeros((64, 64), np.uint8)
    for _ in range(8):
        assert det.update(still, touch=True, mask=mask).slip is False
