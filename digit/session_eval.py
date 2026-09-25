"""Session-level reference construction and honest label loading.

A press-test session records its own *untouched* phase.  The camera's
auto-exposure / white balance can still be settling when ``press-test`` grabs
its start reference (and a ``reference.png`` copied from another run is simply
wrong), so the reference used to evaluate a session is rebuilt from the
session's **own untouched frames** -- the per-pixel median of them.  An
explicit ``--reference`` still overrides this.

Everything here is pure file I/O + NumPy, so it is testable with a small
synthetic session and never touches a sensor.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: session labels are ``index -> (label, cell, phase)``
Labels = Dict[int, Tuple[str, str, str]]


def load_labels(session: str) -> Labels:
    """Read ``labels.csv`` into ``{index: (label, cell, phase)}`` ({} if absent)."""
    path = os.path.join(session, "labels.csv")
    labels: Labels = {}
    if not os.path.isfile(path):
        return labels
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[int(row["index"])] = (
                row.get("label", "unlabeled"),
                row.get("cell", ""),
                row.get("phase", ""),
            )
    return labels


def _frame_index(session: str) -> Dict[int, str]:
    """Read ``frames.csv`` into ``{index: path}``."""
    rows: Dict[int, str] = {}
    path = os.path.join(session, "frames.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[int(row["index"])] = os.path.join(session, row["filename"])
    return rows


def untouched_indices(
    labels: Labels,
    *,
    phase: str = "untouched",
    label: str = "none",
    skip_first: int = 0,
) -> List[int]:
    """Indices of the untouched frames, in capture order.

    A frame counts when its phase is ``untouched`` *or* its label is ``none``.
    ``skip_first`` drops that many leading frames (sensor warm-up).
    """
    idx = [
        i
        for i, (lab, _cell, ph) in sorted(labels.items())
        if ph == phase or lab == label
    ]
    return idx[skip_first:]


def evenly_spaced(items: Sequence[int], n: int) -> List[int]:
    """Pick at most ``n`` items spread evenly over the sequence (keep order)."""
    items = list(items)
    if n <= 0 or len(items) <= n:
        return items
    positions = np.linspace(0, len(items) - 1, n).round().astype(int)
    seen: Dict[int, None] = {}
    for p in positions:
        seen[int(p)] = None
    return [items[p] for p in seen]


def reference_from_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median of a list of BGR frames (uint8)."""
    if not frames:
        raise ValueError("no frames given for the reference")
    stack = np.stack([f.astype(np.float32) for f in frames])
    return np.clip(np.median(stack, axis=0), 0, 255).astype(np.uint8)


def reference_from_untouched(
    session: str,
    labels: Optional[Labels] = None,
    *,
    max_frames: int = 60,
    skip_first: int = 0,
) -> Optional[np.ndarray]:
    """Build a reference as the median of the session's untouched frames.

    Returns ``None`` when the session has no untouched frames or no readable
    frame file.  At most ``max_frames`` evenly spaced untouched frames are
    read, so memory stays bounded on long sessions.
    """
    labels = load_labels(session) if labels is None else labels
    idx = untouched_indices(labels, skip_first=skip_first)
    if not idx:
        return None
    files = _frame_index(session)
    chosen = [files[i] for i in evenly_spaced(idx, max_frames) if i in files]
    frames = []
    for path in chosen:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    if not frames:
        return None
    return reference_from_frames(frames)
