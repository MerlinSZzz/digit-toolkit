"""A synthetic frame source with the same interface as :class:`DigitCamera`.

Purpose: run the live viewer, the monitor and the recorder **without a sensor**
-- for tests, CI, demos and for the case where the gel cannot be touched.  It
animates a contact that moves across the gel, changing shape, so all the live
processing channels have something to show.

It is explicitly a simulation, not sensor data.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from . import synth

#: shapes the animated contact cycles through
KINDS = ["circle", "square", "ridge", "ring", "two_finger"]


class FakeContactSource:
    """Animated synthetic DIGIT frames (drop-in for :class:`DigitCamera`)."""

    def __init__(
        self,
        reference: Optional[np.ndarray] = None,
        fps: int = 30,
        seed: int = 0,
        kinds=None,
        contact_period: float = 6.0,
        motion_period: float = 8.0,
    ) -> None:
        self.reference = (
            reference if reference is not None else synth.synthetic_reference(seed=seed)
        ).astype(np.uint8)
        h, w = self.reference.shape[:2]
        self.width, self.height, self.fps = w, h, int(fps)
        self.led = 15
        self.serial = "FAKE"
        self.dev_name = "fake://synthetic"
        self.revision = "fake"
        self.info = None
        self.orientation = True
        self.kinds = list(kinds or KINDS)
        self.contact_period = float(contact_period)
        self.motion_period = float(motion_period)
        self._rng = np.random.default_rng(seed)
        self._i = 0
        self._t0 = time.time()
        self._cap = None  # set by open() to a truthy sentinel
        self._open = False

    # ------------------------------------------------------------- interface
    def open(self) -> "FakeContactSource":
        self._cap = True
        self._open = True
        self._t0 = time.time()
        return self

    def close(self) -> None:
        self._cap = None
        self._open = False

    def read(self) -> np.ndarray:
        if not self._open:
            raise RuntimeError("fake source is not open")
        i = self._i
        self._i += 1
        h, w = self.reference.shape[:2]
        t = i / max(1.0, self.fps)
        # contact present for most of each contact_period
        phase = (t % self.contact_period) / self.contact_period
        present = phase < 0.75
        # moving centre: a slow lissajous inside the central 60% of the gel
        theta = 2 * np.pi * (t / self.motion_period)
        cx = w * (0.5 + 0.24 * np.sin(theta))
        cy = h * (0.5 + 0.20 * np.sin(2 * theta + 0.7))
        kind = self.kinds[(int(t // self.contact_period)) % len(self.kinds)]
        amplitude = -75.0 if present else 0.0
        frame, _ = synth.synth_frame(
            self.reference,
            kind=kind,
            center=(cx, cy),
            radius=0.13,
            amplitude=amplitude,
            noise=2.0,
            rng=self._rng,
        )
        return frame

    # ------------------------------------------------- convenience / parity
    def set_led(self, level: int) -> int:
        self.led = int(level)
        return self.led

    def set_led_rgb(self, r: int, g: int, b: int) -> int:
        return (int(r) << 8) | (int(g) << 4) | int(b)

    def get_led(self) -> int:
        return self.led

    def measure_fps(self, n: int = 30, timeout: float = 2.0) -> float:
        start = time.perf_counter()
        for _ in range(n):
            self.read()
        elapsed = time.perf_counter() - start
        return n / elapsed if elapsed > 0 else 0.0

    def mode_line(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps} fps (FAKE synthetic source)"

    @property
    def stats(self) -> Dict[str, float]:
        return {"frames": self._i, "elapsed_s": time.time() - self._t0, "mean_fps": self.fps}

    def __enter__(self) -> "FakeContactSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def animated_sequence(
    reference: np.ndarray,
    n: int = 90,
    fps: int = 30,
    seed: int = 0,
    **kwargs,
) -> list:
    """Return ``n`` synthetic frames from a :class:`FakeContactSource`."""
    source = FakeContactSource(reference, fps=fps, seed=seed, **kwargs).open()
    try:
        return [source.read() for _ in range(n)]
    finally:
        source.close()
