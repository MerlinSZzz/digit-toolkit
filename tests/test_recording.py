"""Sensor-free tests for recording / replay."""

import os

import numpy as np

from digit import recording as rec


def _frames(n=4, shape=(48, 64, 3)):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(n)]


def test_session_roundtrip(tmp_path):
    path = str(tmp_path / "sess")
    frames = _frames()
    with rec.SessionWriter(path, serial="TEST", width=64, height=48, fps=30, save_video=False) as writer:
        for f in frames:
            writer.add(f, timestamp_ns=123456789)
    meta = rec.session_meta(path)
    assert meta.n_frames == 4
    assert meta.serial == "TEST"
    loaded = list(rec.iter_session(path))
    assert len(loaded) == 4
    for (_, frame, ts), original in zip(loaded, frames):
        assert ts == 123456789
        assert frame.shape == original.shape
        assert np.array_equal(frame, original)
    assert os.path.isfile(os.path.join(path, "meta.json"))
    assert os.path.isfile(os.path.join(path, "frames.csv"))


def test_session_with_reference(tmp_path):
    path = str(tmp_path / "sess2")
    frames = _frames(2)
    reference = frames[0]
    with rec.SessionWriter(path, width=64, height=48, fps=30, save_video=False, reference=reference) as writer:
        for f in frames:
            writer.add(f)
    assert os.path.isfile(os.path.join(path, "reference.png"))
    loaded_ref = rec.session_meta(path)
    assert loaded_ref.n_frames == 2


def test_list_sessions(tmp_path):
    root = str(tmp_path / "sessions")
    for name in ("a", "b"):
        path = os.path.join(root, name)
        with rec.SessionWriter(path, width=64, height=48, fps=30, save_video=False) as writer:
            writer.add(_frames(1)[0])
    sessions = rec.list_sessions(root)
    assert len(sessions) == 2


def test_export_video(tmp_path):
    path = str(tmp_path / "sess3")
    with rec.SessionWriter(path, width=64, height=48, fps=30, save_video=False) as writer:
        for f in _frames(5):
            writer.add(f)
    out = str(tmp_path / "out.avi")
    rec.export_video(path, out)
    assert os.path.isfile(out) and os.path.getsize(out) > 0
