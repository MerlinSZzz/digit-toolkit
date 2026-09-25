"""Sensor-free tests for the session reference + eval defaults.

These cover the two D3 requirements:

* ``eval-session`` builds its reference from the session's own untouched
  phase, not from a possibly stale ``reference.png``;
* the ``eval-session`` CLI defaults are exactly what the Makefile passes.
"""

from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np
import pytest

from digit import session_eval
from digit import synth
from digit.cli import build_parser, main

MAKEFILE = os.path.join(os.path.dirname(__file__), "..", "Makefile")


def _write_session(path, ref, untouched, press, stale):
    """Write a tiny labelled session folder and return it as ``str``."""
    path = str(path)
    os.makedirs(os.path.join(path, "frames"), exist_ok=True)
    labels = []
    frames = list(untouched) + list(press)
    with open(os.path.join(path, "frames.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "timestamp_ns", "filename"])
        for i, frame in enumerate(frames):
            rel = f"frames/{i:06d}.png"
            cv2.imwrite(os.path.join(path, rel), frame)
            w.writerow([i, i, rel])
            if i < len(untouched):
                labels.append([i, "untouched", "none", ""])
            else:
                labels.append([i, "press-top-center", "press", "top-center"])
    with open(os.path.join(path, "labels.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "phase", "label", "cell"])
        w.writerows(labels)
    cv2.imwrite(os.path.join(path, "reference.png"), stale)
    meta = {"version": 1, "width": ref.shape[1], "height": ref.shape[0],
            "fps": 30, "n_frames": len(frames)}
    with open(os.path.join(path, "protocol.json"), "w", encoding="utf-8") as fh:
        json.dump({"cells": {"top-center": [0.5, 0.25]}, "n_frames": len(frames)}, fh)
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return path


@pytest.fixture
def fake_session(tmp_path):
    ref = synth.synthetic_reference((96, 128, 3), seed=3)
    rng = np.random.default_rng(0)
    untouched = [np.clip(ref.astype(np.int16) + rng.integers(-2, 3, ref.shape), 0, 255).astype(np.uint8)
                 for _ in range(12)]
    press = []
    for _ in range(12):
        frame = ref.copy()
        cv2.circle(frame, (128 // 2, 96 // 4), 12, (0, 0, 0), -1)
        press.append(frame)
    # a deliberately wrong reference (brightness moved, like the real session)
    stale = np.clip(ref.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    return _write_session(tmp_path / "session", ref, untouched, press, stale), ref, stale


def test_reference_is_built_from_untouched_phase(fake_session):
    session, ref, stale = fake_session
    built = session_eval.reference_from_untouched(session)
    assert built is not None
    # near the untouched median / true reference, far from the stale reference.png
    assert np.abs(built.astype(np.int16) - ref.astype(np.int16)).mean() < 4.0
    assert np.abs(built.astype(np.int16) - stale.astype(np.int16)).mean() > 30.0


def test_eval_session_ignores_stale_reference(fake_session, tmp_path):
    session, _ref, _stale = fake_session
    out = str(tmp_path / "eval.json")
    assert main(["eval-session", session, "--no-slip", "--out", out]) == 0
    payload = json.load(open(out, encoding="utf-8"))
    assert payload["reference_source"].startswith("session untouched median")
    # untouched frames are not touch; press frames are
    assert payload["touch"]["fpr_untouched"] == 0.0
    assert payload["touch"]["tpr_press"] == 1.0
    assert payload["min_peak"] == 25.0  # Makefile default, and now in the JSON

    # the stale reference would have made every frame a false positive
    out2 = str(tmp_path / "eval_stale.json")
    assert main(["eval-session", session, "--no-slip", "--reference",
                 os.path.join(session, "reference.png"), "--out", out2]) == 0
    stale_payload = json.load(open(out2, encoding="utf-8"))
    assert stale_payload["touch"]["fpr_untouched"] == 1.0


def _makefile_defaults():
    values = {}
    with open(MAKEFILE, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("THRESHOLD ?="):
                values["threshold"] = float(line.split("?=", 1)[1].strip())
            elif line.startswith("MIN_AREA ?="):
                values["min_area"] = int(line.split("?=", 1)[1].strip())
            elif line.startswith("MIN_PEAK ?="):
                values["min_peak"] = float(line.split("?=", 1)[1].strip())
    return values


def test_eval_session_defaults_match_makefile():
    defaults = _makefile_defaults()
    assert defaults == {"threshold": 10.0, "min_area": 40, "min_peak": 25.0}
    args = build_parser().parse_args(["eval-session"])
    assert args.threshold == defaults["threshold"]
    assert args.min_area == defaults["min_area"]
    assert args.min_peak == defaults["min_peak"]
    # and the Makefile recipe forwards all three
    recipe = open(MAKEFILE, encoding="utf-8").read()
    for flag, name in (("--threshold", "THRESHOLD"), ("--min-area", "MIN_AREA"),
                       ("--min-peak", "MIN_PEAK")):
        assert f"{flag} $({name})" in recipe
