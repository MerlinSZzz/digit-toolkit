"""Recording and replay of DIGIT sessions.

A session is a self-describing folder::

    sessions/20260924_172530/
        meta.json          # sensor, mode, led, count, timestamps
        frames.csv         # index, timestamp_ns, filename
        frames/000000.png  # lossless frames
        video.avi          # optional MJPEG preview for quick browsing
        reference.png      # optional no-contact reference

Frames are stored as PNG so replay is bit-exact and needs no codec.  The AVI
is a convenience preview only.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

SESSION_VERSION = 1


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass
class SessionMeta:
    path: str
    version: int = SESSION_VERSION
    serial: str = ""
    node: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0
    led: Optional[int] = None
    started_at: str = ""
    ended_at: str = ""
    n_frames: int = 0
    tool: str = "digit-toolkit"
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "serial": self.serial,
            "node": self.node,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "led": self.led,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "n_frames": self.n_frames,
            "tool": self.tool,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, path: str, data: Dict[str, object]) -> "SessionMeta":
        known = {f for f in cls.__dataclass_fields__ if f != "path"}  # type: ignore[attr-defined]
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(path=path, **{k: v for k, v in data.items() if k in known}, extra=extra)  # type: ignore[arg-type]


class SessionWriter:
    """Write frames + timestamps into a session folder (context manager)."""

    def __init__(
        self,
        path: str,
        serial: str = "",
        node: str = "",
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        led: Optional[int] = None,
        save_video: bool = True,
        reference: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        self.path = os.path.abspath(path)
        self.frames_dir = os.path.join(self.path, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.meta = SessionMeta(
            path=self.path,
            serial=serial,
            node=node,
            width=int(width),
            height=int(height),
            fps=int(fps),
            led=led,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            extra=dict(extra or {}),
        )
        self.reference = reference
        self._csv = open(os.path.join(self.path, "frames.csv"), "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["index", "timestamp_ns", "filename"])
        self._video: Optional[cv2.VideoWriter] = None
        self.save_video = bool(save_video)
        self._closed = False

    def add(self, frame: np.ndarray, timestamp_ns: Optional[int] = None) -> str:
        if self._closed:
            raise RuntimeError("session already closed")
        index = self.meta.n_frames
        name = f"{index:06d}.png"
        rel = os.path.join("frames", name)
        cv2.imwrite(os.path.join(self.path, rel), frame)
        self._writer.writerow([index, int(timestamp_ns if timestamp_ns is not None else time.time_ns()), rel])
        if self.save_video:
            if self._video is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._video = cv2.VideoWriter(
                    os.path.join(self.path, "video.avi"),
                    fourcc,
                    float(self.meta.fps or 30),
                    (w, h),
                )
            self._video.write(frame)
        self.meta.n_frames += 1
        return rel

    def close(self) -> SessionMeta:
        if self._closed:
            return self.meta
        self._closed = True
        self._csv.close()
        if self._video is not None:
            self._video.release()
            self._video = None
        if self.reference is not None:
            cv2.imwrite(os.path.join(self.path, "reference.png"), self.reference)
        self.meta.ended_at = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta.as_dict(), fh, indent=2, ensure_ascii=False)
        return self.meta

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def record(
    camera,
    outdir: Optional[str] = None,
    seconds: Optional[float] = None,
    frames: Optional[int] = None,
    root: str = "sessions",
    name: Optional[str] = None,
    save_video: bool = True,
    reference: Optional[np.ndarray] = None,
    progress=None,
) -> SessionMeta:
    """Record from a :class:`~digit.capture.DigitCamera` into a session folder.

    Provide ``seconds`` or ``frames`` (both ``None`` records forever).  The
    camera must already be open.
    """
    if outdir is None:
        outdir = os.path.join(root, name or _now_stamp())
    writer = SessionWriter(
        outdir,
        serial=camera.serial or "",
        node=camera.dev_name,
        width=camera.width,
        height=camera.height,
        fps=camera.fps,
        led=camera.led,
        save_video=save_video,
        reference=reference,
    )
    count = 0
    t0 = time.time()
    try:
        while True:
            if frames is not None and count >= frames:
                break
            if seconds is not None and (time.time() - t0) >= seconds:
                break
            frame = camera.read()
            writer.add(frame)
            count += 1
            if progress is not None and count % 10 == 0:
                progress(count, time.time() - t0)
    finally:
        writer.close()
    return writer.meta


def _read_csv(path: str) -> List[Tuple[int, int, str]]:
    rows: List[Tuple[int, int, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((int(row["index"]), int(row["timestamp_ns"]), row["filename"]))
    return rows


def session_meta(path: str) -> SessionMeta:
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
        return SessionMeta.from_dict(path, json.load(fh))


def iter_session(path: str, start: int = 0, stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray, int]]:
    """Yield ``(index, frame, timestamp_ns)`` from a session folder."""
    rows = _read_csv(os.path.join(path, "frames.csv"))
    for index, ts, rel in rows:
        if index < start:
            continue
        if stop is not None and index >= stop:
            break
        frame = cv2.imread(os.path.join(path, rel))
        if frame is None:
            continue
        yield index, frame, ts


def load_session(path: str, step: int = 1) -> Tuple[SessionMeta, List[np.ndarray], List[int]]:
    """Load a whole session into memory (``step`` decimates frames)."""
    frames: List[np.ndarray] = []
    stamps: List[int] = []
    for i, frame, ts in iter_session(path):
        if i % max(1, step) == 0:
            frames.append(frame)
            stamps.append(ts)
    return session_meta(path), frames, stamps


def list_sessions(root: str = "sessions", limit: int = 50) -> List[SessionMeta]:
    """List session folders newest first."""
    if not os.path.isdir(root):
        return []
    paths = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "meta.json"))
    ]
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    out = []
    for p in paths[:limit]:
        try:
            out.append(session_meta(p))
        except Exception:
            continue
    return out


def export_video(path: str, out: str, fps: Optional[float] = None, processing=None) -> str:
    """Re-export a session to an AVI, optionally through a per-frame callable.

    ``processing(frame, index) -> frame`` lets you render a processed video.
    Returns the output path.
    """
    meta = session_meta(path)
    frames = list(iter_session(path))
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    h, w = frames[0][1].shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"MJPG"), float(fps or meta.fps or 30), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out}")
    try:
        for index, frame, _ in frames:
            writer.write(processing(frame, index) if processing else frame)
    finally:
        writer.release()
    return out
