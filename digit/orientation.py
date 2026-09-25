"""Orientation of a DIGIT frame: stored camera view vs the operator's view.

The raw UVC buffer is landscape (640x480).  The official DIGIT convention
(the one ``digit-interface`` uses) is::

    positive = cv2.flip(cv2.transpose(raw), 0)      # -> portrait 480x640

which is the "official" stored view.

The owner holds the sensor with the **rounded end up** (an arrow on the back
points that way) and the **square cable end down**.  Comparing the six
guided-press cells of the 2026-09-25 session against that physical layout shows
that the official portrait frame is **left-right mirrored** relative to the
operator's view: a press at the operator's top-left lands at the *upper right*
of the stored frame, while "up" is already the rounded end (no vertical flip
and no rotation are needed).  The ``operator`` preset therefore flips x only::

    operator = cv2.flip(official, 1)

Use :func:`orient` to apply an explicit ``rotate`` / ``flip_x`` / ``flip_y``
transform, or :func:`apply_preset` with one of :data:`PRESETS`.  ``read()`` on
:class:`~digit.capture.DigitCamera` applies the preset passed as
``orientation=`` (the CLI defaults to ``"operator"``), so the live view and
every processing view show the sensor as the operator holds it.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Union

import cv2
import numpy as np

#: The two named transforms.  Both are applied *after* the camera's official
#: transpose+vertical-flip; ``official`` is the stored convention, ``operator``
#: is the sensor as held (rounded end up, cable down, left/right not mirrored).
PRESETS: Dict[str, Dict[str, object]] = {
    "official": {"rotate": 0, "flip_x": False, "flip_y": False},
    "operator": {"rotate": 0, "flip_x": True, "flip_y": False},
}

#: What live views and processing use when the caller does not say otherwise.
DEFAULT = "operator"

#: session meta key that records the orientation of the stored frames
META_KEY = "frame_orientation"

OrientationSpec = Union[bool, str, Dict[str, object], None]


def resolve(spec: OrientationSpec) -> Dict[str, object]:
    """Normalise an orientation spec to ``{"rotate", "flip_x", "flip_y"}``.

    ``False``/``"raw"``/``None`` -> the identity (sensor-native buffer, no
    official transform either).  ``True``/``"official"`` -> the stored
    portrait convention.  ``"operator"`` -> stored portrait + x mirror.
    A dict is passed through with sensible defaults.
    """
    if spec is None or spec is False or spec == "raw" or spec == "none":
        return {"rotate": 0, "flip_x": False, "flip_y": False, "raw": True}
    if spec is True or spec == "official":
        return dict(PRESETS["official"], raw=False)
    if isinstance(spec, str):
        if spec in PRESETS:
            return dict(PRESETS[spec], raw=False)
        raise ValueError(f"unknown orientation preset {spec!r} (use {sorted(PRESETS)} or 'raw')")
    if isinstance(spec, dict):
        return {
            "rotate": int(spec.get("rotate", 0)) % 360,
            "flip_x": bool(spec.get("flip_x", False)),
            "flip_y": bool(spec.get("flip_y", False)),
            "raw": bool(spec.get("raw", False)),
        }
    raise TypeError(f"bad orientation spec {spec!r}")


def orient(
    frame: np.ndarray,
    rotate: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """Apply ``flip_x``, then ``flip_y``, then ``rotate`` (clockwise degrees).

    ``flip_x`` mirrors left-right (``cv2.flip(..., 1)``); ``flip_y`` mirrors
    top-bottom (``cv2.flip(..., 0)``).  ``rotate`` is one of 0/90/180/270.
    """
    out = frame
    if flip_x:
        out = cv2.flip(out, 1)
    if flip_y:
        out = cv2.flip(out, 0)
    rot = int(rotate) % 360
    if rot == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rot == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot != 0:
        raise ValueError(f"rotate must be 0/90/180/270, got {rotate}")
    return out


def apply_preset(frame: np.ndarray, spec: OrientationSpec = DEFAULT) -> np.ndarray:
    """Apply a named preset / dict / bool to a frame that is already in the
    official portrait view (the camera's transpose+flip)."""
    r = resolve(spec)
    return orient(frame, rotate=int(r["rotate"]), flip_x=bool(r["flip_x"]), flip_y=bool(r["flip_y"]))


# ---------------------------------------------------------------------------
# session-level helpers


def session_frame_orientation(session: str, default: str = "official") -> str:
    """Read ``frame_orientation`` from a session's ``meta.json``.

    Sessions recorded before this key existed stored the **official** portrait
    view, so that is the default.  ``extra`` keys written by ``SessionWriter``
    land at the top level of ``meta.json``.
    """
    path = os.path.join(session, "meta.json")
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return default
    value = data.get(META_KEY)
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return "official" if value else "raw"
    return str(value)


def to_operator(frame: np.ndarray, source: str = "official") -> np.ndarray:
    """Convert a stored frame from ``source`` orientation to the operator view."""
    if source == "operator":
        return frame
    if source in ("official", "raw"):
        return apply_preset(frame, "operator")
    return apply_preset(frame, "operator")
