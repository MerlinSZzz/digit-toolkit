"""Synthetic contact frames built from a real no-contact reference.

The agent building this toolkit cannot physically press the gel, so the
recognition evaluation uses *synthetic* indentation fields added to a real
reference frame.  This module is explicit about that: every generated sample
is ``reference + contact_field + noise``.  It is used by the offline tests and
by ``make eval`` to measure detection / localisation / classification
honestly, and it is also handy as a stand-in when the sensor is unplugged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _coords(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    return xx.astype(np.float32), yy.astype(np.float32)


def super_gaussian(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    sigma: Tuple[float, float],
    power: float = 2.0,
) -> np.ndarray:
    """Elliptical generalised Gaussian in ``0..1`` (power 2 = Gaussian)."""
    xx, yy = _coords(shape)
    cx, cy = center
    sx, sy = max(sigma[0], 1e-3), max(sigma[1], 1e-3)
    r = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
    return np.exp(-(r**power)).astype(np.float32)


def contact_field(
    shape: Tuple[int, int],
    kind: str = "circle",
    center: Optional[Tuple[float, float]] = None,
    radius: float = 0.12,
    amplitude: float = -70.0,
    aspect: float = 1.0,
    angle: float = 0.0,
    power: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """A synthetic indentation field (float, same HxW as ``shape``).

    ``radius`` is a fraction of the image's smaller side.  ``kind`` selects a
    shape whose *support* differs, which is what the classifier has to learn:

    ``circle``, ``square``, ``ridge`` (elongated), ``ring`` (DoG),
    ``two_finger`` (two lobes), ``stripe`` (oriented bars).
    """
    height, width = shape[:2]
    rng = rng or np.random.default_rng(0)
    if center is None:
        center = (width / 2.0, height / 2.0)
    scale = min(height, width)
    sigma = (radius * scale * max(aspect, 0.2), radius * scale / max(aspect, 0.2) * 0.55)

    if kind in ("circle", "square", "ridge"):
        p = {"circle": 2.0, "square": 6.0, "ridge": 2.0}.get(kind, power)
        sx, sy = sigma
        if kind == "ridge":
            sx, sy = sigma[0] * 2.2, sigma[1] * 0.45
        field = super_gaussian(shape, center, (sx, sy), p)
    elif kind == "ring":
        outer = super_gaussian(shape, center, sigma, 2.0)
        inner = super_gaussian(shape, center, (sigma[0] * 0.55, sigma[1] * 0.55), 2.0)
        field = np.clip(outer - inner, 0, 1)
    elif kind == "two_finger":
        d = radius * scale * 0.8
        f1 = super_gaussian(shape, (center[0] - d, center[1]), sigma, 3.0)
        f2 = super_gaussian(shape, (center[0] + d, center[1]), sigma, 3.0)
        field = np.clip(f1 + f2, 0, 1)
    elif kind == "stripe":
        xx, yy = _coords(shape)
        theta = np.deg2rad(angle)
        proj = (xx - center[0]) * np.cos(theta) + (yy - center[1]) * np.sin(theta)
        field = (0.5 + 0.5 * np.sin(proj / (radius * scale * 0.35))) * super_gaussian(
            shape, center, (sigma[0] * 2.5, sigma[1] * 2.5), 2.0
        )
    else:
        raise ValueError(f"unknown contact kind {kind!r}")

    if angle and kind != "stripe":
        field = _rotate(field, angle, center)
    return (field * amplitude).astype(np.float32)


def _rotate(field: np.ndarray, angle_deg: float, center: Tuple[float, float]) -> np.ndarray:
    height, width = field.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, 1.0)
    return cv2.warpAffine(field, matrix, (width, height), flags=cv2.INTER_LINEAR)


def synth_frame(
    reference: np.ndarray,
    kind: str = "circle",
    center: Optional[Tuple[float, float]] = None,
    radius: float = 0.12,
    amplitude: float = -70.0,
    aspect: float = 1.0,
    angle: float = 0.0,
    noise: float = 2.0,
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(frame, true_mask)`` for one synthetic contact.

    The contact is added to all three channels (scaled by ``tint``) so the
    result is a plausible tactile image; ``true_mask`` is the region where the
    field exceeded half its peak -- the ground truth for localisation.
    """
    field = contact_field(
        reference.shape,
        kind=kind,
        center=center,
        radius=radius,
        amplitude=amplitude,
        aspect=aspect,
        angle=angle,
    )
    rng = rng or np.random.default_rng(0)
    frame = reference.astype(np.float32).copy()
    if reference.ndim == 3:
        for c in range(3):
            frame[..., c] += field * float(tint[c])
    else:
        frame += field
    if noise:
        frame += rng.normal(0.0, noise, frame.shape).astype(np.float32)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    support = np.abs(field) > (0.5 * np.abs(field).max()) if field.any() else np.zeros_like(field, bool)
    true_mask = (support.astype(np.uint8)) * 255
    return frame, true_mask


#: default multiclass problem: *where* the contact is
POSITION_CLASSES: Dict[str, Dict[str, object]] = {
    "none": {"kind": "circle", "amplitude": 0.0},
    "center": {"kind": "circle", "center": (0.5, 0.5)},
    "top-left": {"kind": "circle", "center": (0.32, 0.32)},
    "top-right": {"kind": "circle", "center": (0.68, 0.32)},
    "bottom-left": {"kind": "circle", "center": (0.32, 0.68)},
    "bottom-right": {"kind": "circle", "center": (0.68, 0.68)},
}

#: alternative problem: *what shape / object* is touching
SHAPE_CLASSES: Dict[str, Dict[str, object]] = {
    "none": {"kind": "circle", "amplitude": 0.0},
    "blunt": {"kind": "circle", "radius": 0.16, "amplitude": -55.0},
    "sharp": {"kind": "square", "radius": 0.10, "amplitude": -80.0},
    "edge": {"kind": "ridge", "radius": 0.15, "amplitude": -60.0, "aspect": 2.2},
    "ring": {"kind": "ring", "radius": 0.15, "amplitude": -70.0},
    "two-finger": {"kind": "two_finger", "radius": 0.10, "amplitude": -70.0},
}


def synthesize_dataset(
    reference: np.ndarray,
    classes: Optional[Dict[str, Dict[str, object]]] = None,
    n_per_class: int = 8,
    seed: int = 0,
    jitter: float = 0.04,
    amplitude_jitter: float = 0.25,
) -> Tuple[List[np.ndarray], List[str], List[Dict[str, object]]]:
    """Generate a balanced synthetic dataset around one reference frame.

    Each class gets ``n_per_class`` samples with a small random position,
    radius and amplitude jitter.  Returns ``(frames, labels, infos)``.
    """
    classes = classes or POSITION_CLASSES
    height, width = reference.shape[:2]
    frames: List[np.ndarray] = []
    labels: List[str] = []
    infos: List[Dict[str, object]] = []
    for ci, (label, cls_params) in enumerate(classes.items()):
        for i in range(n_per_class):
            rng = np.random.default_rng(seed * 1000 + ci * 100 + i)
            params = dict(cls_params)
            center = params.pop("center", None)
            if center is not None:
                center = (
                    float(center[0]) * width + rng.uniform(-jitter, jitter) * width,
                    float(center[1]) * height + rng.uniform(-jitter, jitter) * height,
                )
            amp = params.get("amplitude", -70.0)
            if amp:
                params["amplitude"] = float(amp) * (1.0 + rng.uniform(-amplitude_jitter, amplitude_jitter))
            params.setdefault("radius", 0.12)  # type: ignore[arg-type]
            params.setdefault("noise", 2.0)  # type: ignore[arg-type]
            frame, mask = synth_frame(reference, center=center, rng=rng, **params)  # type: ignore[arg-type]
            frames.append(frame)
            labels.append(label)
            infos.append(
                {
                    "label": label,
                    "center": center,
                    "true_mask": mask,
                    "params": {k: v for k, v in params.items() if k != "noise"},
                }
            )
    return frames, labels, infos


def true_centroid(info: Dict[str, object]) -> Optional[Tuple[float, float]]:
    """Ground-truth centroid of a synthetic sample, if it has one."""
    mask = info.get("true_mask")
    if mask is None or not np.asarray(mask).any():  # type: ignore[union-attr]
        return None
    ys, xs = np.nonzero(np.asarray(mask))
    return (float(xs.mean()), float(ys.mean()))


def synthetic_reference(shape: Tuple[int, int, int] = (480, 640, 3), seed: int = 0) -> np.ndarray:
    """A plausible DIGIT-like no-contact gel image, for sensor-free demos/tests.

    It is a smooth, softly-lit surface with low-frequency colour variation,
    a gentle vignette and a little texture -- enough for the processing code to
    have a realistic reference to subtract.
    """
    height, width = shape[:2]
    rng = np.random.default_rng(seed)
    base = np.zeros(shape, np.float32)
    for c in range(shape[2] if shape[2] == 3 else 1):
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx = rng.uniform(0.3, 0.7) * width
        cy = rng.uniform(0.3, 0.7) * height
        sigma = rng.uniform(0.4, 0.9)
        blob = np.exp(-(((xx - cx) / (sigma * width)) ** 2 + ((yy - cy) / (sigma * height)) ** 2))
        # a bright, fairly uniform base so the whole image is usable gel
        base[..., c] = 120 + 24 * blob
    texture = rng.normal(0, 2.0, (height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.5)
    if shape[2] == 3:
        for c in range(3):
            base[..., c] += texture
    # gentle vignette: keep even the corners above the active-region threshold
    vignette = 1.0 - 0.15 * (
        ((np.linspace(-1, 1, width))[None, :] ** 2 + (np.linspace(-1, 1, height))[:, None] ** 2) / 2.0
    )
    base *= vignette[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)

