"""Frame capture and LED control for the DIGIT.

Wraps :class:`cv2.VideoCapture` with three behaviours the raw OpenCV object
does not give you:

* it opens the **capture** node, never the metadata node;
* it applies the official DIGIT orientation (``transpose`` + vertical flip),
  so the returned image is the upright portrait view ``(H=W, W=H)``;
* LED intensity is set through the UVC "Zoom" control using the same 12-bit
  RGB packing as the official ``digit-interface`` package, and every mode
  change is verified by reading a frame.

No ``digit-interface`` object is instantiated, because its ``connect()``
hard-codes QVGA @ 60 fps and can leave the camera mid-stream if it fails;
the packing and orientation rules are reused instead (credit in README).
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .device import DigitInfo, find_digit

#: UVC "Zoom, Absolute" control is reused by DIGIT firmware for the LEDs.
LED_MIN = 0
LED_MAX = 15


class CameraError(RuntimeError):
    """Raised when the sensor cannot be opened or a frame cannot be read."""


def _pack_rgb(r: int, g: int, b: int) -> int:
    for value in (r, g, b):
        if not 0 <= int(value) <= LED_MAX:
            raise ValueError(f"LED values must be 0..{LED_MAX}, got {value}")
    return (int(r) << 8) | (int(g) << 4) | int(b)


class DigitCamera:
    """Blocking, single-threaded DIGIT camera.

    Example
    -------
    >>> with DigitCamera() as cam:          # doctest: +SKIP
    ...     cam.set_led(15)
    ...     frame = cam.read()              # BGR uint8, official orientation
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        node: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        orientation: bool = True,
        led: Optional[int] = LED_MAX,
        warmup: int = 8,
        buffersize: int = 1,
    ) -> None:
        self.serial = serial
        self.node = node
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.orientation = bool(orientation)
        self.led = led
        self.warmup = int(warmup)
        self.buffersize = int(buffersize)

        self.info: Optional[DigitInfo] = None
        self.dev_name: str = node or ""
        self.revision: str = ""
        self._cap: Optional[cv2.VideoCapture] = None
        self._frames = 0
        self._t_first = 0.0
        self._t_last = 0.0
        self._last_shape: Tuple[int, ...] = ()
        self._modes: Optional[list] = None

    # ------------------------------------------------------------------ open
    def open(self) -> "DigitCamera":
        if self._cap is not None:
            return self
        if self.node is None:
            self.info = find_digit(self.serial)
            self.dev_name = self.info.dev_name
        else:
            try:
                self.info = find_digit(self.serial, node=self.node)
                self.dev_name = self.info.dev_name
            except RuntimeError:
                self.info = DigitInfo(dev_name=self.node, serial=self.serial or "unknown")
                self.dev_name = self.node
        if self.info is not None:
            self.revision = self.info.revision

        cap = cv2.VideoCapture(self.dev_name, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"could not open {self.dev_name}. Is another program using the "
                f"sensor? Do you have video-group permission (`id`)?"
            )
        self._cap = cap
        try:
            self._configure(self.width, self.height, self.fps)
            if self.led is not None:
                self.set_led(self.led)
            for _ in range(max(0, self.warmup)):
                self._read_raw()
        except Exception:
            self.close()
            raise
        self._t_first = self._t_last = time.time()
        return self

    # ------------------------------------------------------------------ read
    def _read_raw(self) -> np.ndarray:
        if self._cap is None:
            raise CameraError("camera is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError(
                f"no frame from {self.dev_name} (USB glitch or sensor wedged). "
                f"Try `make reset` or a physical replug."
            )
        return frame

    def read(self) -> np.ndarray:
        """Return one BGR frame in the official orientation."""
        frame = self._read_raw()
        self._last_shape = frame.shape
        if self.orientation:
            frame = cv2.flip(cv2.transpose(frame), 0)
        self._frames += 1
        self._t_last = time.time()
        return frame

    # --------------------------------------------------------------- controls
    def set_led(self, level: int) -> int:
        """Set all three LED channels to ``level`` (0..15)."""
        return self.set_led_rgb(level, level, level)

    def set_led_rgb(self, r: int, g: int, b: int) -> int:
        """Set the red/green/blue LED channels independently (0..15 each)."""
        if self._cap is None:
            raise CameraError("camera is not open")
        packed = _pack_rgb(r, g, b)
        self._cap.set(cv2.CAP_PROP_ZOOM, packed)
        self.led = int(r)  # single-channel level is only meaningful for sets
        return packed

    def get_led(self) -> Optional[int]:
        """Read back the packed LED value, if the driver reports it."""
        if self._cap is None:
            return None
        value = self._cap.get(cv2.CAP_PROP_ZOOM)
        return None if value is None else int(value)

    def set_mode(self, width: int, height: int, fps: int) -> Tuple[int, int, int]:
        """Change resolution/fps safely.

        The DIGIT/UVC combo does **not** like being reconfigured while it is
        already streaming (a mid-stream fps change wedged this unit once), so
        this closes and reopens the camera instead of poking the live handle.
        """
        width, height, fps = int(width), int(height), int(fps)
        if (width, height, fps) != (self.width, self.height, self.fps):
            self.width, self.height, self.fps = width, height, fps
            if self._cap is not None:
                self.close()
                self.open()
        return self.width, self.height, self.fps

    def _configure(self, width: int, height: int, fps: int) -> Tuple[int, int, int]:
        """Apply a mode to a freshly opened handle (never mid-stream)."""
        if self._cap is None:
            raise CameraError("camera is not open")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self._cap.set(cv2.CAP_PROP_FPS, int(fps))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffersize)
        read_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        read_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        read_fps = int(round(self._cap.get(cv2.CAP_PROP_FPS)))
        try:
            self._read_raw()
        except CameraError:
            if (width, height) != (640, 480):
                # fall back to the safest full-resolution mode
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap.set(cv2.CAP_PROP_FPS, 30)
                self._read_raw()
                read_w, read_h, read_fps = 640, 480, 30
            else:
                raise
        self.width, self.height, self.fps = read_w, read_h, read_fps
        return read_w, read_h, read_fps

    def set_fps(self, fps: int) -> int:
        return self.set_mode(self.width, self.height, fps)[2]

    def reopen(self) -> "DigitCamera":
        """Close and reopen the camera (use after a USB glitch)."""
        self.close()
        time.sleep(0.5)
        return self.open()

    # -------------------------------------------------------------- benchmark
    def measure_fps(self, n: int = 60, timeout: float = 4.0) -> float:
        """Grab ``n`` frames as fast as possible and return the achieved fps."""
        if self._cap is None:
            raise CameraError("camera is not open")
        # drain one stale buffer
        self._read_raw()
        count = 0
        start = time.perf_counter()
        while count < n and (time.perf_counter() - start) < timeout:
            self._read_raw()
            count += 1
        elapsed = time.perf_counter() - start
        return count / elapsed if elapsed > 0 else 0.0

    @property
    def stats(self) -> Dict[str, float]:
        elapsed = max(1e-9, self._t_last - self._t_first)
        return {
            "frames": self._frames,
            "elapsed_s": elapsed,
            "mean_fps": self._frames / elapsed if self._frames else 0.0,
        }

    def mode_line(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps} fps "
            f"(sensor-native {self._last_shape[1]}x{self._last_shape[0]}) "
            f"led={self.led} node={self.dev_name}"
        )

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def __enter__(self) -> "DigitCamera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        self.close()


class CaptureThread(threading.Thread):
    """Background reader that always keeps the newest frame available.

    Consumers call :meth:`wait_new` with the last sequence number they saw.
    On a read error the thread marks itself disconnected and tries to reopen
    the camera, so a loose USB cable does not kill a live view.
    """

    def __init__(self, camera: DigitCamera, reconnect: bool = True) -> None:
        super().__init__(daemon=True)
        self.camera = camera
        self.reconnect = reconnect

        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._running = True
        self._connected = False
        self._error: Optional[str] = None
        self._fps = 0.0

    # --------------------------------------------------------------- threading
    def run(self) -> None:  # pragma: no cover - timing dependent
        while self._running:
            try:
                if self.camera._cap is None:
                    self.camera.open()
                frame = self.camera.read()
            except Exception as exc:  # noqa: BLE001 - report and retry
                with self._cond:
                    self._connected = False
                    self._error = str(exc)
                    self._cond.notify_all()
                if not self.reconnect:
                    break
                time.sleep(0.5)
                try:
                    self.camera.close()
                    self.camera.open()
                except Exception:
                    time.sleep(1.0)
                continue
            now = time.time()
            with self._cond:
                if self._frame is not None:
                    dt = now - getattr(self, "_t_prev", now)
                    inst = 1.0 / dt if dt > 0 else 0.0
                    self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
                self._t_prev = now
                self._frame = frame
                self._seq += 1
                self._connected = True
                self._error = None
                self._cond.notify_all()

    # ------------------------------------------------------------------- API
    def wait_new(
        self, last_seq: int, timeout: float = 1.0
    ) -> Tuple[Optional[np.ndarray], int]:
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq > last_seq or not self._running, timeout=timeout
            )
            if self._seq > last_seq and self._frame is not None:
                return self._frame.copy(), self._seq
            return None, last_seq

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def seq(self) -> int:
        return self._seq

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
