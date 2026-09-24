"""Sensor-free tests for capture safety and the YUYV roll fix.

The mode-change tests use a **fake OpenCV capture** that records every mode
property set *after* the first read.  A live-stream property change is exactly
what wedged the real DIGIT once, so the fake turns that into a failing test.
"""

import cv2
import numpy as np
import pytest

from digit import capture as C


class FakeVideoCapture:
    """A stand-in for ``cv2.VideoCapture`` that watches for live mode changes."""

    def __init__(self, name=None, backend=None):
        self.name = name
        self.opened = True
        self.released = False
        self.streaming = False
        self.reads = 0
        self.mode_sets_while_streaming = []
        self.props = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_BUFFERSIZE: 1.0,
        }

    def isOpened(self):
        return self.opened and not self.released

    def set(self, prop, value):
        if self.streaming and prop in C.MODE_PROPERTIES:
            self.mode_sets_while_streaming.append((prop, float(value)))
        self.props[prop] = float(value)
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        self.streaming = True
        self.reads += 1
        w = int(self.props.get(cv2.CAP_PROP_FRAME_WIDTH, 640))
        h = int(self.props.get(cv2.CAP_PROP_FRAME_HEIGHT, 480))
        return True, np.full((h, w, 3), 100, np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_captures(monkeypatch):
    created = []

    def factory(name=None, backend=None):
        fake = FakeVideoCapture(name, backend)
        created.append(fake)
        return fake

    monkeypatch.setattr(C.cv2, "VideoCapture", factory)
    monkeypatch.setattr(C, "find_digit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fake")))
    return created


def test_mode_change_reopens_instead_of_poking_live_stream(fake_captures):
    cam = C.DigitCamera(node="/dev/fake", width=640, height=480, fps=30, warmup=0)
    cam.open()
    assert len(fake_captures) == 1
    cam.read()  # the stream is now running

    cam.set_fps(15)  # a mid-view mode change

    # the old handle must never have received a mode property while streaming
    assert fake_captures[0].mode_sets_while_streaming == []
    # ... and a fresh handle was opened with the new mode set before its read
    assert len(fake_captures) == 2
    assert fake_captures[1].mode_sets_while_streaming == []
    assert fake_captures[1].props[cv2.CAP_PROP_FPS] == 15.0
    assert fake_captures[1].reads >= 1
    assert (cam.width, cam.height, cam.fps) == (640, 480, 15)
    cam.close()
    assert fake_captures[1].released


def test_resolution_change_reopens_with_new_size(fake_captures):
    cam = C.DigitCamera(node="/dev/fake", width=640, height=480, fps=30, warmup=0)
    cam.open()
    cam.read()
    cam.set_mode(320, 240, 30)
    assert fake_captures[0].mode_sets_while_streaming == []
    assert len(fake_captures) == 2
    assert fake_captures[1].props[cv2.CAP_PROP_FRAME_WIDTH] == 320.0
    assert fake_captures[1].props[cv2.CAP_PROP_FRAME_HEIGHT] == 240.0
    assert (cam.width, cam.height) == (320, 240)
    cam.close()


def test_configure_refuses_to_change_a_live_stream(fake_captures):
    cam = C.DigitCamera(node="/dev/fake", width=640, height=480, fps=30, warmup=0)
    cam.open()
    cam.read()
    assert cam._stream_open is True
    with pytest.raises(C.CameraError):
        cam._configure(320, 240, 30)
    cam.close()
    assert cam._stream_open is False


# ---------------------------------------------------------------------------
# roll fix


def _textured(h=240, w=320, seed=0):
    """A smooth, gel-like image with a gentle horizontal ramp.

    Low-frequency content is what makes a *wrap* seam stand out: neighbouring
    columns agree, but the two distant columns that a roll brings together do
    not.  (White noise would make every column boundary equally sharp.)
    """
    rng = np.random.default_rng(seed)
    small = rng.integers(80, 180, size=(max(2, h // 24), max(2, w // 24)), dtype=np.uint8)
    base = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=4.0)
    ramp = np.linspace(-40.0, 40.0, w, dtype=np.float32)[None, :]
    base = np.clip(base + ramp, 0, 255).astype(np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def test_detect_roll_finds_and_fixes_a_wrapped_buffer():
    clean = _textured()
    shift = 37
    rolled = np.roll(clean, shift, axis=1)
    found, confidence = C.detect_roll(rolled)
    assert found == -shift  # the correction that undoes a +37 roll is -37
    assert confidence > C.ROLL_MIN_CONFIDENCE
    fixed, applied, _ = C.fix_roll(rolled)
    assert applied == -shift
    # a roll is lossless, so un-rolling recovers the clean frame exactly
    assert np.array_equal(fixed, clean)


def test_detect_roll_leaves_a_clean_frame_alone():
    clean = _textured(seed=3)
    found, _ = C.detect_roll(clean)
    assert found == 0


def test_detect_roll_leaves_a_physical_pan_alone():
    # a real sideways move does not wrap: both crops are continuous, so there
    # is no seam to find.
    big = _textured(h=240, w=420, seed=5)
    crop_a = big[:, :320]
    crop_b = big[:, 60:380]
    assert C.detect_roll(crop_a)[0] == 0
    assert C.detect_roll(crop_b)[0] == 0

