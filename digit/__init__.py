"""DIGIT tactile sensor toolkit.

A small, beginner-friendly toolkit for the Facebook/Meta **DIGIT** optical
tactile sensor (USB ``2833:0209``).  It provides device discovery, capture at
the sensor's real V4L2 modes, LED control, reference/difference processing,
contact detection and localisation, a relative depth/normal reconstruction,
optical-flow / slip sensing, recording and replay, and a classical recognition
example.

The package is deliberately import-light: the core modules only need ``numpy``
and ``opencv-python``.  Optional extras:

* ``pyudev`` (via ``digit-interface``)   -- richer USB/udev device discovery
* ``linuxpy``                            -- exact V4L2 mode/control enumeration
* ``scipy``                              -- DCT Poisson depth integration
* ``scikit-learn``                       -- learned recognition example

Public entry points::

    from digit import DigitCamera, find_digit, supported_modes
    from digit import processing, recognition, recording, flow

Command line::

    python -m digit --help
"""

from __future__ import annotations

__version__ = "0.1.0"

from .device import (  # noqa: F401
    DEFAULT_SERIAL,
    DigitInfo,
    find_digit,
    list_digits,
    supported_modes,
)
from .capture import DigitCamera, CaptureThread, CameraError  # noqa: F401

__all__ = [
    "__version__",
    "DEFAULT_SERIAL",
    "DigitInfo",
    "find_digit",
    "list_digits",
    "supported_modes",
    "DigitCamera",
    "CaptureThread",
    "CameraError",
]
