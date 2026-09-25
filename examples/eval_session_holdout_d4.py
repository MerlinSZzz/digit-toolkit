#!/usr/bin/env python3
"""Honest hold-out evaluation of a real ``press-test`` session (D4).

Sensor-free companion for the report.  Run it as::

    python examples/eval_session_holdout_d4.py sessions/press_20260925_092909 \
        --out .logs/holdout_d4.json

It rebuilds the reference from the **tuning half** of the session's own
untouched frames, splits every label into two alternating halves, analyses the
stored (official portrait) frames and **reports in the operator view** (sensor
as held: rounded end up, cable down, not mirrored), tunes on half A and reports
on half B: localisation per cell (a retuned brightness mask and a
lighting-invariant deformation method), the orientation check across the four
portrait transforms, and slip events per label.
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
from digit import orientation as orientation_mod  # noqa: E402
from digit import processing as P  # noqa: E402
from digit import session_eval  # noqa: E402


def split_halves(indices: List[int]) -> Tuple[List[int], List[int]]:
    """Alternate by capture order: even rank -> A, odd rank -> B."""
    return ([i for r, i in enumerate(indices) if r % 2 == 0],
            [i for r, i in enumerate(indices) if r % 2 == 1])


def mean_or_none(values):
    return round(statistics.fmean(values), 1) if values else None


def median_or_none(values):
    return round(statistics.median(values), 1) if values else None


def _transforms(width: int, height: int):
    return {
        "identity": lambda x, y: (x, y),
        "flip_x": lambda x, y: (width - 1 - x, y),
        "flip_y": lambda x, y: (x, height - 1 - y),
        "rot180": lambda x, y: (width - 1 - x, height - 1 - y),
    }


def _layout_ok(points: Dict[str, Tuple[float, float]]) -> bool:
    need = ("top-left", "top-center", "top-right", "middle-left", "middle-right", "bottom-center")
    if any(k not in points for k in need):
        return False
    tl, tc, tr = points["top-left"], points["top-center"], points["top-right"]
    ml, mr = points["middle-left"], points["middle-right"]
    bc = points["bottom-center"]
    top_y = (tl[1] + tc[1] + tr[1]) / 3.0
    mid_y = (ml[1] + mr[1]) / 2.0
    return (top_y < mid_y < bc[1]
            and tl[0] < tc[0] < tr[0] and ml[0] < mr[0]
            and abs(tl[0] - ml[0]) < 120 and abs(tr[0] - mr[0]) < 120
            and abs(tc[0] - bc[0]) < 120)


class BaselineLocaliser:
    """D3-style brightness mask centroid (max-channel |frame-reference|)."""

    def __init__(self, magnitude: np.ndarray, region: np.ndarray):
        self.mag = magnitude
        self.region = region

    def predict(self, thr: float, min_area: int, min_peak: float):
        m = (self.mag > float(thr)).astype(np.uint8) * 255
        m = cv2.bitwise_and(m, (self.region > 0).astype(np.uint8) * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= int(min_area):
                keep[labels == k] = 255
        ys, xs = np.nonzero(keep)
        if xs.size == 0 or float(self.mag[keep > 0].max()) < float(min_peak):
            return None
        return (float(xs.mean()), float(ys.mean()))


class DeformationLocaliser:
    """Lighting-invariant high-pass deformation localiser."""

    def __init__(self, deformation: np.ndarray, region: np.ndarray):
        self.dmap = deformation
        self.region = region

    def predict(self, quantile: float):
        return P.deformation_centroid(self.dmap, region=self.region, quantile=float(quantile))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="subsample per half (default all)")
    args = ap.parse_args()

    labels = session_eval.load_labels(args.session)
    proto_path = os.path.join(args.session, "protocol.json")
    proto = json.load(open(proto_path, encoding="utf-8")) if os.path.isfile(proto_path) else {"cells": {}}
    cells: Dict[str, List[float]] = proto.get("cells", {})

    stored_orientation = orientation_mod.session_frame_orientation(args.session)
    print(f"session stored orientation: {stored_orientation} -> operator view is flip_x")

    by_label: Dict[str, List[int]] = {}
    for i, (lab, _cell, _ph) in sorted(labels.items()):
        by_label.setdefault(lab, []).append(i)
    halves = {lab: split_halves(idx) for lab, idx in by_label.items()}
    if args.max_frames:
        halves = {lab: (a[: args.max_frames], b[: args.max_frames]) for lab, (a, b) in halves.items()}

    files = session_eval._frame_index(args.session)

    def read(i: int) -> Optional[np.ndarray]:
        # stored (official portrait) frame; the operator view is applied when a
        # prediction is mapped for reporting (flip_x) or in the orientation check.
        return cv2.imread(files[i])

    unt_a = [i for i, (_l, _c, ph) in labels.items()
             if ph == "untouched" and i in set(halves.get("none", ([], []))[0])]
    chosen = session_eval.evenly_spaced(unt_a, 60)
    ref = session_eval.reference_from_frames([f for f in (read(i) for i in chosen) if f is not None])
    region = P.active_region(ref)
    height, width = ref.shape[:2]
    print(f"reference from {len(chosen)} untouched tuning frames; frame {width}x{height}")

    cell_centers = {name: (cx * width, cy * height) for name, (cx, cy) in cells.items()}
    transforms = _transforms(width, height)

    press_a = halves.get("press", ([], []))[0]
    press_b = halves.get("press", ([], []))[1]
    press_by_cell_a: Dict[str, List[int]] = {}
    press_by_cell_b: Dict[str, List[int]] = {}
    for i in press_a:
        press_by_cell_a.setdefault(labels[i][1], []).append(i)
    for i in press_b:
        press_by_cell_b.setdefault(labels[i][1], []).append(i)

    mags = {}
    dmaps = {}
    for i in sorted(set(press_a) | set(press_b)):
        img = read(i)
        signed = img.astype(np.float32) - ref.astype(np.float32)
        mags[i] = P.magnitude(signed)
        dmaps[i] = P.deformation_map(img, ref, sigma=40.0, post_sigma=30.0)

    def points_for_baseline(indices, params):
        loc = BaselineLocaliser(np.zeros((1, 1), np.float32), region)
        out = []
        for i in indices:
            loc.mag = mags[i]
            p = loc.predict(*params)
            if p is not None:
                out.append(p)
        return out

    def points_for_deform(indices, quantile):
        loc = DeformationLocaliser(np.zeros((1, 1), np.float32), region)
        out = []
        for i in indices:
            loc.dmap = dmaps[i]
            p = loc.predict(quantile)
            if p is not None:
                out.append(p)
        return out

    best = None
    for thr in (10, 15, 20, 25, 30, 35):
        for ma in (30, 100, 300):
            for mp in (0.0, 25.0, 45.0):
                errs = []
                for cell, idxs in press_by_cell_a.items():
                    for p in points_for_baseline(idxs, (thr, ma, mp)):
                        x, y = transforms["flip_x"](*p)
                        nx, ny = cell_centers[cell]
                        errs.append(float(np.hypot(x - nx, y - ny)))
                if errs and (best is None or statistics.fmean(errs) < best[0]):
                    best = (statistics.fmean(errs), (thr, ma, mp))
    bparams = best[1]
    print(f"baseline tuned {bparams} (half A mean {best[0]:.1f} px)")

    bestd = None
    for q in (0.2, 0.3, 0.4, 0.5, 0.6):
        errs = []
        for cell, idxs in press_by_cell_a.items():
            for p in points_for_deform(idxs, q):
                x, y = transforms["flip_x"](*p)
                nx, ny = cell_centers[cell]
                errs.append(float(np.hypot(x - nx, y - ny)))
        if errs and (bestd is None or statistics.fmean(errs) < bestd[0]):
            bestd = (statistics.fmean(errs), q)
    dq = bestd[1]
    print(f"deformation quantile tuned {dq} (half A mean {bestd[0]:.1f} px)")

    def report(points_by_cell, tag):
        per_cell = {}
        all_err = []
        correct = total = 0
        for cell, pts in points_by_cell.items():
            ee = []
            for (x, y) in pts:
                nx, ny = cell_centers[cell]
                ee.append(float(np.hypot(x - nx, y - ny)))
                near = min(cell_centers, key=lambda c: (x - cell_centers[c][0]) ** 2 + (y - cell_centers[c][1]) ** 2)
                correct += int(near == cell)
                total += 1
            per_cell[cell] = {"mean_px": mean_or_none(ee), "median_px": median_or_none(ee), "n": len(ee)}
            all_err += ee
        payload = {"overall_mean_px": mean_or_none(all_err), "overall_median_px": median_or_none(all_err),
                   "cell_accuracy": round(correct / max(1, total), 3), "n": total, "per_cell": per_cell}
        print(f"{tag}: overall {payload['overall_mean_px']} px, median {payload['overall_median_px']} px, "
              f"cell acc {payload['cell_accuracy']} (n={total})")
        return payload

    base_pts = {}
    deform_pts = {}
    for cell, idxs in press_by_cell_b.items():
        base_pts[cell] = [transforms["flip_x"](*p) for p in points_for_baseline(idxs, bparams)]
        deform_pts[cell] = [transforms["flip_x"](*p) for p in points_for_deform(idxs, dq)]
    baseline_report = report(base_pts, "baseline (retuned brightness mask)")
    deform_report = report(deform_pts, "deformation (lighting-invariant)")

    orientation_check = {}
    for name, T in transforms.items():
        pts = {}
        errs = []
        for cell, idxs in press_by_cell_b.items():
            cc = [T(*p) for p in points_for_deform(idxs, dq)]
            if cc:
                pts[cell] = (float(np.mean([p[0] for p in cc])), float(np.mean([p[1] for p in cc])))
                for (x, y) in cc:
                    nx, ny = cell_centers[cell]
                    errs.append(float(np.hypot(x - nx, y - ny)))
        orientation_check[name] = {
            "mean_px": mean_or_none(errs),
            "layout_ok": _layout_ok(pts),
            "mean_points": {k: [round(v[0], 1), round(v[1], 1)] for k, v in pts.items()},
        }
        print(f"orientation {name:9s}: mean {orientation_check[name]['mean_px']} px, layout_ok "
              f"{orientation_check[name]['layout_ok']}")

    # Slip is gated on the shipped/validated touch detector (the D3 held-out
    # touch numbers: press/object TPR 1.00, untouched FPR 0.00), separate from
    # the localisation-mask params that may be tuned for a different objective.
    TOUCH_GATE = {"threshold": 10.0, "min_area": 40, "min_peak": 25.0}
    touch_thr = TOUCH_GATE["threshold"]
    touch_area = TOUCH_GATE["min_area"]
    touch_peak = TOUCH_GATE["min_peak"]

    order = sorted(labels)
    gray_cache: Dict[int, np.ndarray] = {}
    mask_cache: Dict[int, Optional[np.ndarray]] = {}
    touch_cache: Dict[int, bool] = {}
    for i in order:
        img = read(i)
        if img is None:
            continue
        gray_cache[i] = P.to_gray(img)
        signed = img.astype(np.float32) - ref.astype(np.float32)
        mask = P.contact_mask(signed, threshold=touch_thr, min_area=touch_area, region=region)
        if mask.sum() == 0:
            touch_cache[i], mask_cache[i] = False, mask
        else:
            mag = P.magnitude(signed)
            touch_cache[i], mask_cache[i] = bool(mag[mask > 0].max() >= touch_peak), mask

    def stable(indices, guard=5):
        if len(indices) <= 2 * guard:
            return list(indices)
        return list(indices[guard:-guard])

    phase_frames: Dict[str, List[int]] = {}
    for ph in sorted({labels[i][2] for i in order}):
        phase_frames[ph] = [i for i in order if labels[i][2] == ph]
    label_half_b: Dict[str, set] = {}
    for lab in ("none", "press", "slide", "object"):
        idx = [i for i in order if labels[i][0] == lab]
        label_half_b[lab] = set(split_halves(idx)[1])

    def stable_set(label):
        if label not in ("press", "object"):
            return set(label_half_b[label])
        keep = set()
        for ph, ph_idx in phase_frames.items():
            if not ph_idx or labels[ph_idx[0]][0] != label:
                continue
            keep.update(stable(ph_idx))
        return set(i for i in label_half_b[label] if i in keep)

    def run_events(threshold, use_mask):
        """Run the detector over each label's consecutive frames; return the
        frame index where each slip event starts, per label."""
        out = {}
        for lab in ("none", "press", "slide", "object"):
            full = [i for i in order if labels[i][0] == lab]
            det = flow_mod.TextureSlipDetector(threshold=threshold, required=3, lag=5)
            starts = []
            slip_set = set()
            run = 0
            for i in full:
                if i not in gray_cache:
                    continue
                ev = det.update(
                    gray_cache[i],
                    touch_cache[i],
                    mask=(mask_cache[i] if use_mask else None),
                    region=region,
                )
                if ev.slip:
                    slip_set.add(i)
                    run += 1
                    if run == 1:
                        starts.append(i)
                else:
                    run = 0
            out[lab] = {"starts": starts, "slip_set": slip_set, "frames": len(full)}
        return out

    def count_on(events, indices):
        s = set(indices)
        return sum(1 for i in events if i in s)

    methods = {}
    for use_mask, mname in ((True, "inside_mask"), (False, "gel_region")):
        threshold = 1.6
        for T in (1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2):
            ev = run_events(T, use_mask)
            pe = count_on(ev["press"]["starts"], stable_set("press"))
            se = count_on(ev["slide"]["starts"], label_half_b["slide"])
            if pe == 0 and se > 0:
                threshold = T
                break
        ev = run_events(threshold, use_mask)
        per_label = {}
        for lab in ("none", "press", "slide", "object"):
            held = label_half_b[lab]
            per_label[lab] = {
                "held_out_frames": len(held),
                "events": count_on(ev[lab]["starts"], held),
                "slip_frames": len(ev[lab]["slip_set"] & held),
                "stable_middle": {
                    "frames": len(stable_set(lab)),
                    "events": count_on(ev[lab]["starts"], stable_set(lab)),
                    "slip_frames": len(ev[lab]["slip_set"] & stable_set(lab)),
                },
            }
        methods[mname] = {"threshold": threshold, "per_label": per_label}
        print(f"slip[{mname}] threshold={threshold}: " +
              " ".join(
                  f"{lab}={per_label[lab]['events']}ev/{per_label[lab]['slip_frames']}fr" for lab in per_label))

    payload = {
        "session": os.path.abspath(args.session),
        "stored_orientation": stored_orientation,
        "operator_view": "x mirror of the stored official portrait (rounded end up)",
        "reference": "median of untouched tuning-half frames",
        "tuned_on": "half A (alternating frames per label)",
        "baseline_params": {"threshold": bparams[0], "min_area": bparams[1], "min_peak": bparams[2]},
        "deformation_params": {"sigma": 40.0, "post_sigma": 30.0, "quantile": dq},
        "localisation": {"baseline": baseline_report, "deformation": deform_report},
        "orientation_check": orientation_check,
        "slip": {"params": {"required": 3, "lag": 5, "touch_gate": TOUCH_GATE},
                 "methods": methods},
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "orientation_check"}, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
