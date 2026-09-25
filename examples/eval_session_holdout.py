#!/usr/bin/env python3
"""Honest hold-out evaluation of a real ``press-test`` session.

The DIGIT toolkit's ``make eval-session`` reports on every frame.  This script
is the *honest* companion the report asks for:

* it rebuilds the reference from the session's own untouched frames (the
  per-pixel median of the **tuning half** only);
* it splits every label's frames into two halves (alternating by capture
  order, so both halves span the whole phase);
* it tunes ``threshold`` / ``min_area`` / ``min_peak`` on half A;
* it reports touch TPR (press, object) and FPR (untouched) on the held-out
  half B, per-cell localisation error, and slip on the full slide phase.

It is sensor-free and reads a session folder.  Run it as::

    python examples/eval_session_holdout.py .logs/session_real --out .logs/holdout.json

All quantities are relative to the cell centres stored in ``protocol.json``
(normalised ``(x/W, y/H)`` in the stored portrait frame).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digit import flow as flow_mod  # noqa: E402
from digit import processing as P  # noqa: E402
from digit import session_eval  # noqa: E402


def mask_from_mag(
    mag: np.ndarray,
    threshold: float,
    min_area: int,
    region: Optional[np.ndarray],
    open_ksize: int = 3,
    close_ksize: int = 7,
) -> np.ndarray:
    mask = (mag > float(threshold)).astype(np.uint8) * 255
    if region is not None:
        mask = cv2.bitwise_and(mask, region)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))
    if min_area and min_area > 0:
        n, labels = cv2.connectedComponents(mask, 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            comp = labels == i
            if int(comp.sum()) >= int(min_area):
                keep[comp] = 255
        mask = keep
    return mask


def mask_summary(mask: np.ndarray, mag: np.ndarray) -> Tuple[int, Optional[Tuple[float, float]], float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, None, 0.0
    area = int(xs.size)
    centroid = (float(xs.mean()), float(ys.mean()))
    peak = float(mag[mask > 0].max())
    return area, centroid, peak


def split_halves(indices: List[int]) -> Tuple[List[int], List[int]]:
    """Alternate by capture order: even rank -> A, odd rank -> B."""
    a = [i for r, i in enumerate(indices) if r % 2 == 0]
    b = [i for r, i in enumerate(indices) if r % 2 == 1]
    return a, b


def grid(thresholds, min_areas, min_peaks):
    for t in thresholds:
        for a in min_areas:
            for p in min_peaks:
                yield float(t), int(a), float(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="subsample per half (default all)")
    args = ap.parse_args()

    labels = session_eval.load_labels(args.session)
    meta = json.load(open(os.path.join(args.session, "meta.json"), encoding="utf-8"))
    proto = json.load(open(os.path.join(args.session, "protocol.json"), encoding="utf-8"))
    cells: Dict[str, List[float]] = proto["cells"]

    by_label: Dict[str, List[int]] = {}
    for i, (lab, _cell, _ph) in sorted(labels.items()):
        by_label.setdefault(lab, []).append(i)
    halves = {lab: split_halves(idx) for lab, idx in by_label.items()}
    if args.max_frames:
        halves = {
            lab: (a[: args.max_frames], b[: args.max_frames]) for lab, (a, b) in halves.items()
        }

    # reference from the tuning half's untouched frames only (no leakage)
    unt_a = [i for i, (_l, _c, ph) in labels.items() if ph == "untouched" and i in set(halves["none"][0])]
    files = session_eval._frame_index(args.session)
    chosen = session_eval.evenly_spaced(unt_a, 60)
    ref_frames = [cv2.imread(files[i]) for i in chosen if i in files]
    ref = session_eval.reference_from_frames([f for f in ref_frames if f is not None])
    region = P.active_region(ref)
    height, width = ref.shape[:2]
    print(f"reference from {len(ref_frames)} untouched tuning frames; frame {width}x{height}")

    # precompute magnitude for every frame we may touch
    all_idx = sorted(set(sum([a for a, _ in halves.values()], []) + sum([b for _, b in halves.values()], [])))
    mags: Dict[int, np.ndarray] = {}
    for i in all_idx:
        img = cv2.imread(files[i])
        if img is None:
            continue
        signed = img.astype(np.float32) - ref.astype(np.float32)
        mags[i] = P.magnitude(signed).astype(np.uint8)
    print(f"cached magnitude for {len(mags)} frames")

    thresholds = [8, 10, 12, 15, 18, 22, 26, 30, 35]
    min_areas = [30, 60, 100, 200]
    min_peaks = [0, 10, 15, 20, 25, 30, 35, 40, 50]

    # cache per (threshold, min_area): frame -> (area, centroid, peak)
    cache: Dict[Tuple[float, int], Dict[int, Tuple[int, Optional[Tuple[float, float]], float]]] = {}
    for t in thresholds:
        for a in min_areas:
            d: Dict[int, Tuple[int, Optional[Tuple[float, float]], float]] = {}
            for i, mag in mags.items():
                m = mask_from_mag(mag, t, a, region)
                d[i] = mask_summary(m, mag)
            cache[(float(t), int(a))] = d
    print("mask cache built")

    def metrics_on(indices: List[int], t: float, a: int, p: float, label: str):
        d = cache[(t, a)]
        touch = sum(1 for i in indices if d[i][0] > 0 and d[i][2] >= p)
        return touch / max(1, len(indices)), touch, len(indices)

    # tune on half A
    best = None
    for t, a, p in grid(thresholds, min_areas, min_peaks):
        tpr_press, _tp, _n = metrics_on(halves["press"][0], t, a, p, "press")
        tpr_obj, _to, _no = metrics_on(halves["object"][0], t, a, p, "object")
        fpr, _fp, _nn = metrics_on(halves["none"][0], t, a, p, "none")
        feasible = tpr_press >= 0.95 and tpr_obj >= 0.95
        score = (tpr_press + tpr_obj) / 2.0 - 2.0 * fpr
        # prefer feasible, then lowest FPR, then highest threshold, then highest min_peak
        key = (0 if feasible else 1, fpr, -t, -p) if feasible else (1, -score, -t, -p)
        if best is None or key < best[0]:
            best = (key, t, a, p, {"tpr_press": tpr_press, "tpr_object": tpr_obj, "fpr_none": fpr})
    _key, bt, ba, bp, tuneA = best

    # held-out half B
    held = {}
    for lab in ("press", "object", "none", "slide"):
        if lab not in halves:
            continue
        d = cache[(bt, ba)]
        tr, touch, n = metrics_on(halves[lab][1], bt, ba, bp, lab)
        held[lab] = {
            "frames": n,
            "touch_frames": touch,
            "touch_ratio": round(tr, 4),
            "mean_area_px": round(
                statistics.fmean([d[i][0] for i in halves[lab][1] if d[i][0] > 0]), 1
            )
            if any(d[i][0] > 0 for i in halves[lab][1])
            else 0.0,
        }

    # localisation on held-out press frames, raw and x-mirrored protocol centres
    per_cell: Dict[str, Dict[str, object]] = {}
    all_err_raw: List[float] = []
    all_err_mirror: List[float] = []
    cell_centers = {name: (cx * width, cy * height) for name, (cx, cy) in cells.items()}
    for name in cells:
        idxs = [i for i in halves["press"][1] if labels[i][1] == name]
        d = cache[(bt, ba)]
        raw, mirror = [], []
        for i in idxs:
            area, centroid, peak = d[i]
            if area <= 0 or centroid is None or peak < bp:
                continue
            cx, cy = cell_centers[name]
            raw.append(float(np.hypot(centroid[0] - cx, centroid[1] - cy)))
            mx = width - cx
            mirror.append(float(np.hypot(centroid[0] - mx, centroid[1] - cy)))
        all_err_raw += raw
        all_err_mirror += mirror
        per_cell[name] = {
            "frames": len(idxs),
            "loc_n": len(raw),
            "raw_mean_px": round(statistics.fmean(raw), 1) if raw else None,
            "mirror_mean_px": round(statistics.fmean(mirror), 1) if mirror else None,
        }

    # slip on the full contiguous phases (dense flow + SlipDetector)
    slip = {}
    slip_detector_args = dict(shear_threshold=0.6, centroid_threshold=1.5, required=3)
    for name, phase in (("untouched", "untouched"), ("press", "press"), ("slide", "slide")):
        if phase == "press":
            idxs = [i for i, (_l, _c, ph) in sorted(labels.items()) if ph.startswith("press-")]
        else:
            idxs = [i for i, (_l, _c, ph) in sorted(labels.items()) if ph == phase]
        if not idxs:
            continue
        det = flow_mod.SlipDetector(**slip_detector_args)
        prev_gray = None
        events = 0
        frames = 0
        prev_slip = False
        for i in idxs:
            img = cv2.imread(files[i])
            if img is None:
                continue
            signed = img.astype(np.float32) - ref.astype(np.float32)
            m = mask_from_mag(P.magnitude(signed).astype(np.uint8), bt, ba, region)
            gray = P.to_gray(img)
            if prev_gray is not None:
                flow = flow_mod.dense_flow(prev_gray, gray)
                shear = float(flow_mod.shear_from_dense(flow, m).get("magnitude", 0.0))
                ys, xs = np.nonzero(m)
                centroid = (float(xs.mean()), float(ys.mean())) if xs.size else None
                ev = det.update(centroid, shear)
                if ev.slip:
                    frames += 1
                    if not prev_slip:
                        events += 1
                prev_slip = ev.slip
            prev_gray = gray
        slip[name] = {"frames": len(idxs), "slip_frames": frames, "slip_events": events}

    payload = {
        "session": os.path.abspath(args.session),
        "reference": "median of untouched tuning-half frames",
        "tuned_on": "half A (alternating frames per label)",
        "params": {"threshold": bt, "min_area": ba, "min_peak": bp},
        "tuning_half_a": tuneA,
        "held_out_half_b": held,
        "localisation": {
            "n": len(all_err_raw),
            "raw_mean_px": round(statistics.fmean(all_err_raw), 1) if all_err_raw else None,
            "raw_p90_px": round(float(np.percentile(all_err_raw, 90)), 1) if all_err_raw else None,
            "mirror_mean_px": round(statistics.fmean(all_err_mirror), 1) if all_err_mirror else None,
            "per_cell": per_cell,
        },
        "slip": slip,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
