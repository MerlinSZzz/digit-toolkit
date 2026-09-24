"""Optical-flow and slip sensing for the DIGIT.

Dense Farneback flow gives a textured motion field; sparse Lucas-Kanade on
gel texture gives a cheap, robust shear estimate.  "Slip" is inferred when the
contact region moves relative to the background, or when the contact centroid
accelerates -- the same signal PyTouch and the GelSight handover monitor use.
All magnitudes are in pixels unless stated otherwise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .processing import to_gray


def dense_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> np.ndarray:
    """Farneback dense optical flow, shape ``(H, W, 2)`` (dx, dy)."""
    if prev_gray.ndim == 3:
        prev_gray = to_gray(prev_gray)
    if curr_gray.ndim == 3:
        curr_gray = to_gray(curr_gray)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        0,
    )


def flow_vis(flow: np.ndarray, max_magnitude: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(hsv_bgr, magnitude)`` for a flow field.

    Direction is hue (HSV wheel), magnitude is value.  ``max_magnitude`` sets
    the value scale; when ``None`` it is auto-scaled to the 99th percentile of
    the magnitude (floor 0.5 px), so small flows stay visible.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if max_magnitude is None:
        max_magnitude = max(0.5, float(np.percentile(mag, 99)))
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max(max_magnitude, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), mag


def sparse_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_corners: int = 250,
    quality: float = 0.01,
    min_distance: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade displacement of gel-texture corners.

    Returns ``(prev_points, curr_points, status)`` where arrays have shape
    ``(N, 1, 2)`` and ``(N, 1)`` exactly as OpenCV returns them.
    """
    if prev_gray.ndim == 3:
        prev_gray = to_gray(prev_gray)
    if curr_gray.ndim == 3:
        curr_gray = to_gray(curr_gray)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    if mask is not None:
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    else:
        mask = np.ones_like(prev_gray) * 255
    p0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance, mask=mask
    )
    if p0 is None:
        return (
            np.zeros((0, 1, 2), np.float32),
            np.zeros((0, 1, 2), np.float32),
            np.zeros((0, 1), np.uint8),
        )
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    return p0, p1, status


def shear_from_sparse(
    p0: np.ndarray,
    p1: np.ndarray,
    status: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Median displacement inside the contact minus outside (background).

    Subtracting the background cancels global camera/mount motion, which is
    what makes this a usable shear signal on a hand-held sensor.
    """
    if len(p0) == 0 or status is None or status.sum() == 0:
        return {"n": 0, "n_load": 0, "dx": 0.0, "dy": 0.0, "magnitude": 0.0, "quality": 0.0}
    good = status.reshape(-1) == 1
    prev = p0.reshape(-1, 2)[good]
    disp = (p1.reshape(-1, 2)[good] - prev).astype(np.float64)
    if mask is None:
        tx, ty = np.median(disp[:, 0]), np.median(disp[:, 1])
        return {
            "n": int(len(disp)),
            "n_load": int(len(disp)),
            "dx": float(tx),
            "dy": float(ty),
            "magnitude": float(np.hypot(tx, ty)),
            "quality": 1.0,
        }
    h, w = mask.shape[:2]
    xs = np.clip(prev[:, 0].astype(int), 0, w - 1)
    ys = np.clip(prev[:, 1].astype(int), 0, h - 1)
    inside = mask[ys, xs] > 0
    n_load = int(inside.sum())
    if n_load < 3 or (~inside).sum() < 3:
        tx = float(np.median(disp[:, 0])) if len(disp) else 0.0
        ty = float(np.median(disp[:, 1])) if len(disp) else 0.0
        return {
            "n": int(len(disp)),
            "n_load": n_load,
            "dx": tx,
            "dy": ty,
            "magnitude": float(np.hypot(tx, ty)),
            "quality": 0.0,
        }
    load = np.median(disp[inside], axis=0)
    background = np.median(disp[~inside], axis=0)
    shear = load - background
    return {
        "n": int(len(disp)),
        "n_load": n_load,
        "dx": float(shear[0]),
        "dy": float(shear[1]),
        "magnitude": float(np.hypot(shear[0], shear[1])),
        "quality": float(n_load) / max(1, len(disp)),
    }


def shear_from_dense(flow: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Median dense-flow magnitude inside the contact, background-subtracted."""
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if mask is None or not mask.any():
        inside = mag.reshape(-1)
    else:
        inside = mag[mask > 0]
        outside = mag[mask == 0]
        if inside.size and outside.size:
            return {
                "magnitude": float(np.median(inside) - np.median(outside)),
                "inside": float(np.median(inside)),
                "outside": float(np.median(outside)),
            }
    return {"magnitude": float(np.median(inside)), "inside": float(np.median(inside)), "outside": 0.0}


@dataclass
class SlipEvent:
    slip: bool
    magnitude: float
    centroid_speed: float
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "slip": bool(self.slip),
            "magnitude": round(float(self.magnitude), 3),
            "centroid_speed": round(float(self.centroid_speed), 3),
            "reason": self.reason,
        }


class SlipDetector:
    """Small temporal state machine for slip detection.

    Two independent cues are combined:

    * **shear** -- contact-region optical flow relative to the background
      (``shear_threshold`` px/frame);
    * **centroid motion** -- the contact centroid moving faster than
      ``centroid_threshold`` px/frame (e.g. an object sliding out of grip).

    ``required`` consecutive frames above threshold must agree, which suppresses
    single-frame noise.  It reports rather than actuates; tune the thresholds.
    """

    def __init__(
        self,
        shear_threshold: float = 0.6,
        centroid_threshold: float = 1.5,
        required: int = 3,
        history: int = 30,
    ) -> None:
        self.shear_threshold = float(shear_threshold)
        self.centroid_threshold = float(centroid_threshold)
        self.required = int(required)
        self._hits: Deque[str] = deque(maxlen=max(1, required))
        self._centroid_history: Deque[Optional[Tuple[float, float]]] = deque(maxlen=history)

    def update(
        self,
        centroid: Optional[Tuple[float, float]],
        shear_magnitude: float,
    ) -> SlipEvent:
        speed = 0.0
        if centroid is not None and self._centroid_history and self._centroid_history[-1] is not None:
            px, py = self._centroid_history[-1]  # type: ignore[misc]
            speed = float(np.hypot(centroid[0] - px, centroid[1] - py))
        self._centroid_history.append(centroid)

        reason = ""
        if shear_magnitude >= self.shear_threshold:
            reason = "shear"
        elif speed >= self.centroid_threshold:
            reason = "centroid"
        self._hits.append(reason)
        good = sum(1 for r in self._hits if r) >= self.required
        return SlipEvent(
            slip=bool(good),
            magnitude=float(shear_magnitude),
            centroid_speed=speed,
            reason=reason or "stable",
        )
