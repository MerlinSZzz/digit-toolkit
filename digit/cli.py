"""Command-line interface: ``python -m digit <command> [options]``.

Every ``make`` target in the Makefile is a thin wrapper around one of these
commands, so the same features are available interactively.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import __version__
from . import flow as flow_mod
from . import processing as P
from . import recognition as R
from . import recording as rec
from . import synth
from . import viz
from .capture import CameraError, DigitCamera, CaptureThread
from .device import describe, find_digit, list_digits, supported_modes
from .live import capture_reference, load_reference, run_view

# ---------------------------------------------------------------------------
# helpers


def _log(msg: str) -> None:
    print(msg, flush=True)


def _camera_from_args(args, open_now: bool = True):
    if getattr(args, "fake", False):
        from .fake import FakeContactSource

        reference = None
        ref_path = getattr(args, "fake_reference", None)
        if ref_path:
            reference = load_reference(ref_path)
        camera = FakeContactSource(reference=reference, fps=int(getattr(args, "fps", 30) or 30))
        if open_now:
            camera.open()
        return camera
    width, height = 640, 480
    if getattr(args, "resolution", None):
        width, height = _parse_resolution(args.resolution)
    camera = DigitCamera(
        serial=getattr(args, "serial", None),
        node=getattr(args, "node", None),
        width=width,
        height=height,
        fps=int(getattr(args, "fps", 30) or 30),
        led=getattr(args, "led", 15),
        orientation=not getattr(args, "raw", False),
    )
    if open_now:
        camera.open()
    return camera


def _parse_resolution(text: str) -> Tuple[int, int]:
    for sep in ("x", "X", ",", ":"):
        if sep in text:
            a, b = text.split(sep, 1)
            return int(a), int(b)
    raise argparse.ArgumentTypeError(f"resolution must be WxH, got {text!r}")


def _save_json(path: str, payload: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def _reference_for(args) -> np.ndarray:
    """Load a reference from disk, or capture one, or build a synthetic one."""
    if getattr(args, "reference", None):
        return load_reference(args.reference)
    if getattr(args, "synthetic_reference", False) or getattr(args, "fake", False):
        return synth.synthetic_reference()
    camera = _camera_from_args(args)
    try:
        return capture_reference(camera, n=int(getattr(args, "ref_frames", 15)))
    finally:
        camera.close()


# ---------------------------------------------------------------------------
# commands


def cmd_list(args) -> int:
    devices = list_digits()
    if not devices:
        _log("no DIGIT video nodes found (lsusb should show 2833:0209)")
        return 1
    for dev in devices:
        _log(describe(dev))
        if dev.by_id:
            _log(f"    stable path: {dev.by_id}")
    return 0


def cmd_info(args) -> int:
    try:
        dev = find_digit(args.serial)
    except RuntimeError as exc:
        _log(f"ERROR: {exc}")
        return 1
    _log(f"serial      : {dev.serial}")
    _log(f"node        : {dev.dev_name}")
    _log(f"manufacturer: {dev.manufacturer}")
    _log(f"model       : {dev.model}")
    _log(f"revision    : {dev.revision}")
    if dev.by_id:
        _log(f"stable path : {dev.by_id}")
    _log("supported modes:")
    for mode in supported_modes(dev.dev_name):
        _log(f"  {mode['width']}x{mode['height']} @ {mode['fps']} fps  ({mode['pixel_format']})")
    return 0


def cmd_modes(args) -> int:
    for mode in supported_modes(getattr(args, "node", None)):
        _log(f"{mode['width']}x{mode['height']} @ {mode['fps']} fps  {mode['pixel_format']}  {mode['description']}")
    return 0


def cmd_check(args) -> int:
    from .device import usb_device_path

    ok = True
    _log("== DIGIT sensor check ==")
    try:
        dev = find_digit(args.serial, node=getattr(args, "node", None))
    except RuntimeError as exc:
        _log(f"[FAIL] device discovery: {exc}")
        return 1
    _log(f"[ ok ] sensor found: {describe(dev)}")
    usb = usb_device_path(dev.serial)
    _log(f"[ ok ] usb node: {usb or 'not visible in /sys'}")
    modes = supported_modes(dev.dev_name)
    _log(f"[ ok ] supported modes: " + ", ".join(f"{m['width']}x{m['height']}@{m['fps']}" for m in modes))
    try:
        camera = _camera_from_args(args)
    except CameraError as exc:
        _log(f"[FAIL] open: {exc}")
        return 1
    try:
        fps = camera.measure_fps(n=max(30, int(args.frames)))
        frame = camera.read()
        stats = camera.stats
        region = P.active_region(frame)
        _log(f"[ ok ] mode            : {camera.mode_line()}")
        _log(f"[ ok ] measured fps    : {fps:.1f} (read {int(stats['frames'])} frames)")
        _log(f"[ ok ] frame           : shape={frame.shape} dtype={frame.dtype} mean={frame.mean():.1f} min={frame.min()} max={frame.max()}")
        _log(f"[ ok ] usable gel area : {100.0 * region.mean() / 255.0:.1f}% of pixels")
        if camera.led is not None:
            _log(f"[ ok ] led             : level={camera.led} packed={camera.get_led()}")
        out = args.out or os.path.join(".logs", "check_frame.png")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        cv2.imwrite(out, frame)
        _log(f"[ ok ] saved frame     : {out}")
    finally:
        camera.close()
    _log("== all checks passed ==" if ok else "== checks failed ==")
    return 0 if ok else 1


def crop_region_mean(frame: np.ndarray, region: np.ndarray) -> np.ndarray:
    bbox = P.region_bbox(region)
    if bbox is None:
        return frame
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w]


def cmd_snapshot(args) -> int:
    camera = _camera_from_args(args)
    try:
        for _ in range(int(args.warmup)):
            camera.read()
        frame = camera.read()
    finally:
        camera.close()
    out = args.out or os.path.join("snapshots", time.strftime("snapshot_%Y%m%d_%H%M%S.png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, frame)
    _log(f"snapshot saved: {out}  shape={frame.shape}  {camera.mode_line()}")
    return 0


def cmd_reference(args) -> int:
    camera = _camera_from_args(args)
    try:
        for _ in range(int(args.warmup)):
            camera.read()
        ref = capture_reference(camera, n=int(args.frames))
    finally:
        camera.close()
    out = args.out or "reference.png"
    cv2.imwrite(out, ref)
    np.save(os.path.splitext(out)[0] + ".npy", ref)
    region = P.active_region(ref)
    _log(f"reference saved: {out} (+ .npy)  mean={ref.mean():.1f}  gel area={100.0*region.mean()/255.0:.1f}%")
    return 0


def cmd_led(args) -> int:
    camera = _camera_from_args(args)
    try:
        if args.rgb is not None:
            packed = camera.set_led_rgb(*[int(v) for v in args.rgb])
            _log(f"LED RGB set to {tuple(args.rgb)} (packed=0x{packed:04x})")
        else:
            packed = camera.set_led(int(args.level))
            _log(f"LED level set to {args.level} (packed=0x{packed:04x})")
        time.sleep(0.3)
        frame = camera.read()
        b, g, r = [float(frame[..., i].mean()) for i in range(3)]
        _log(
            f"frame means B={b:.1f} G={g:.1f} R={r:.1f} (use these to confirm the LED; "
            f"CAP_PROP_ZOOM readback={camera.get_led()} is clamped on this unit)"
        )
    finally:
        camera.close()
    return 0


def cmd_fps(args) -> int:
    width, height = _parse_resolution(args.resolution) if args.resolution else (640, 480)
    camera = DigitCamera(
        serial=args.serial,
        node=getattr(args, "node", None),
        width=width,
        height=height,
        fps=int(args.fps),
        led=getattr(args, "led", 15),
        orientation=not getattr(args, "raw", False),
    )
    try:
        camera.open()
        measured = camera.measure_fps(30)
        _log(f"requested fps={args.fps} -> mode {camera.width}x{camera.height}@{camera.fps}, measured {measured:.1f} fps")
    finally:
        camera.close()
    return 0


def cmd_resolution(args) -> int:
    width, height = _parse_resolution(args.resolution)
    camera = DigitCamera(
        serial=args.serial,
        node=getattr(args, "node", None),
        width=width,
        height=height,
        fps=int(args.fps),
        led=getattr(args, "led", 15),
        orientation=not getattr(args, "raw", False),
    )
    try:
        camera.open()
        measured = camera.measure_fps(30)
        _log(f"requested {args.resolution}@{args.fps} -> mode {camera.width}x{camera.height}@{camera.fps}, measured {measured:.1f} fps")
    finally:
        camera.close()
    return 0


def cmd_benchmark(args) -> int:
    rows: List[Dict[str, object]] = []
    for mode in supported_modes(getattr(args, "node", None)):
        try:
            camera = DigitCamera(
                serial=args.serial,
                node=getattr(args, "node", None),
                width=int(mode["width"]),
                height=int(mode["height"]),
                fps=int(mode["fps"]),
            )
            camera.open()
            measured = camera.measure_fps(n=int(args.frames))
            frame = camera.read()
            rows.append(
                {
                    "requested": f"{mode['width']}x{mode['height']}@{mode['fps']}",
                    "actual": f"{camera.width}x{camera.height}@{camera.fps}",
                    "measured_fps": round(float(measured), 2),
                    "frame_shape": list(frame.shape),
                    "mean": round(float(frame.mean()), 1),
                }
            )
            camera.close()
        except Exception as exc:  # noqa: BLE001
            rows.append({"requested": f"{mode['width']}x{mode['height']}@{mode['fps']}", "error": str(exc)})
        time.sleep(0.3)
    for row in rows:
        _log(str(row))
    if args.out:
        _save_json(args.out, {"modes": rows})
        _log(f"saved {args.out}")
    return 0


def cmd_view(args) -> int:
    camera = _camera_from_args(args)
    reference = None
    if args.reference:
        reference = load_reference(args.reference)
    try:
        summary = run_view(
            camera,
            mode=args.mode,
            reference=reference,
            out=args.out,
            frames=args.frames,
            seconds=args.seconds,
            show=args.show,
            threshold=args.threshold,
            min_area=args.min_area,
            diff_gain=args.diff_gain,
            slip=not args.no_slip,
            record_dir=args.record,
            save_video=args.video,
            snapshot_dir=args.snapshot_dir,
            status_every=args.status_every,
        )
    finally:
        camera.close()
    _log(f"view finished: {summary}")
    return 0


def cmd_record(args) -> int:
    camera = _camera_from_args(args)
    reference = load_reference(args.reference) if args.reference else getattr(camera, "reference", None)
    try:
        meta = rec.record(
            camera,
            outdir=args.out,
            seconds=args.seconds,
            frames=args.frames,
            root=args.root,
            name=args.name,
            save_video=not args.no_video,
            reference=reference,
            progress=None if args.quiet else (lambda n, t: None),
        )
    finally:
        camera.close()
    _log(f"recorded {meta.n_frames} frames -> {meta.path}")
    return 0


def cmd_sessions(args) -> int:
    sessions = rec.list_sessions(args.root, limit=args.limit)
    if not sessions:
        _log(f"no sessions under {os.path.abspath(args.root)}")
        return 0
    for s in sessions:
        _log(f"{s.path}  {s.width}x{s.height}@{s.fps}  frames={s.n_frames}  serial={s.serial}  {s.started_at}")
    return 0


def cmd_play(args) -> int:
    path = args.session
    if not os.path.isdir(path):
        sessions = rec.list_sessions(args.root, limit=1)
        if not sessions:
            _log(f"session not found: {path}")
            return 1
        path = sessions[0].path
    meta = rec.session_meta(path)
    _log(f"session {path}: {meta.n_frames} frames {meta.width}x{meta.height}@{meta.fps} serial={meta.serial}")
    reference = load_reference(args.reference) if args.reference else None
    if reference is None:
        ref_path = os.path.join(path, "reference.png")
        if os.path.isfile(ref_path):
            reference = cv2.imread(ref_path)
    processor = (
        P.TactileProcessor(reference, threshold=args.threshold, min_area=args.min_area, depth=args.process == "depth")
        if reference is not None
        else None
    )

    def render(frame: np.ndarray, index: int) -> np.ndarray:
        if processor is None:
            return frame
        tactile = processor.process(frame)
        if args.process == "depth":
            return P.colorize_depth(tactile.depth if tactile.depth is not None else P.relative_height(frame, reference))
        if args.process == "contact":
            return P.draw_contact(P.overlay_mask(frame, tactile.mask), tactile.stats)
        if args.process == "diff":
            return tactile.diff_vis
        return viz.dashboard(tactile)

    if args.out:
        if args.out.lower().endswith((".avi", ".mp4", ".mkv")):
            rec.export_video(path, args.out, fps=args.fps, processing=render if processor else None)
            _log(f"wrote video {args.out}")
            return 0
        if args.out.lower().endswith((".png", ".jpg", ".jpeg")):
            tiles = []
            for index, frame, _ in rec.iter_session(path):
                if index % max(1, args.step) != 0:
                    continue
                tiles.append(P.label_above(viz.fit_tile(render(frame, index)), f"frame {index}"))
                if len(tiles) >= 12:
                    break
            if not tiles:
                _log("no frames to montage")
                return 1
            cv2.imwrite(args.out, viz.stack_grid(tiles, cols=4))
            _log(f"wrote montage {args.out} ({len(tiles)} frames)")
            return 0
        os.makedirs(args.out, exist_ok=True)
        for index, frame, _ in rec.iter_session(path):
            if index % max(1, args.step) != 0:
                continue
            cv2.imwrite(os.path.join(args.out, f"{index:06d}.png"), render(frame, index))
        _log(f"wrote frames to {args.out}")
        return 0

    if args.show:
        for index, frame, ts in rec.iter_session(path, start=args.start, stop=args.stop):
            if index % max(1, args.step) != 0:
                continue
            cv2.imshow("DIGIT replay", render(frame, index))
            if cv2.waitKey(int(1000 / max(1, args.fps or meta.fps or 30))) & 0xFF in (27, ord("q")):
                break
        cv2.destroyAllWindows()
        return 0

    # default: numeric summary + stats if a reference is available
    stamps = []
    for _, _, ts in rec.iter_session(path):
        stamps.append(ts)
    info: Dict[str, object] = {
        "path": path,
        "n_frames": meta.n_frames,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "duration_s": round((max(stamps) - min(stamps)) / 1e9, 3) if len(stamps) > 1 else 0.0,
    }
    if processor is not None:
        touches = 0
        areas = []
        for _, frame, _ in rec.iter_session(path):
            st = processor.process(frame).stats
            touches += int(st.touch)
            areas.append(st.area_px)
        info["touch_frames"] = touches
        info["touch_ratio"] = round(touches / max(1, meta.n_frames), 3)
        info["mean_area_px"] = round(float(np.mean(areas)), 1)
    _log(json.dumps(info, indent=2))
    if args.out_json:
        _save_json(args.out_json, info)
    return 0


def cmd_monitor(args) -> int:
    camera = _camera_from_args(args)
    reference = load_reference(args.reference) if args.reference else None
    try:
        if reference is None:
            reference = capture_reference(camera, n=args.ref_frames)
        detector = R.TouchDetector(reference, threshold=args.threshold, min_area=args.min_area)
        slip_detector = flow_mod.SlipDetector()
        thread = CaptureThread(camera)
        thread.start()
        _, last = thread.wait_new(-1, timeout=3.0)
        t0 = time.time()
        prev_gray = None
        last_print = 0.0
        try:
            while True:
                if args.seconds and time.time() - t0 >= args.seconds:
                    break
                if args.frames and last >= args.frames:
                    break
                frame, last = thread.wait_new(last, timeout=1.0)
                if frame is None:
                    continue
                result = detector.detect(frame)
                shear = 0.0
                if prev_gray is not None:
                    fl = flow_mod.dense_flow(prev_gray, P.to_gray(frame))
                    shear = float(flow_mod.shear_from_dense(fl, None).get("magnitude", 0.0))
                prev_gray = P.to_gray(frame).copy()
                event = slip_detector.update(result.centroid, shear)
                state = {
                    "t": round(time.time() - t0, 3),
                    "touch": result.touch,
                    "area_px": result.area_px,
                    "area_frac": round(result.area_frac, 4),
                    "centroid": result.centroid,
                    "cell": result.grid_cell,
                    "force": round(result.force, 1),
                    "slip": event.slip,
                    "shear": round(event.magnitude, 3),
                    "fps": round(thread.fps, 1),
                }
                if args.json:
                    _log(json.dumps(state))
                elif time.time() - last_print >= args.interval:
                    _log(
                        f"touch={str(result.touch):5s} area={result.area_px:5d}px force={result.force:8.0f} "
                        f"cell={str(result.grid_cell):12s} slip={event.slip} shear={event.magnitude:.2f}px fps={thread.fps:.1f}"
                    )
                    last_print = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            thread.stop()
            thread.join(timeout=1.0)
    finally:
        camera.close()
    return 0


def _reference_for_eval(args) -> Tuple[np.ndarray, str]:
    if args.reference:
        return load_reference(args.reference), f"file:{args.reference}"
    if args.fake or args.synthetic_reference:
        return synth.synthetic_reference(), "synthetic"
    try:
        camera = _camera_from_args(args)
    except Exception as exc:  # noqa: BLE001
        _log(f"no sensor ({exc}); using a synthetic reference instead")
        return synth.synthetic_reference(), "synthetic"
    try:
        return capture_reference(camera, n=args.ref_frames, verbose=False), "sensor"
    finally:
        camera.close()


def cmd_eval(args) -> int:
    reference, source = _reference_for_eval(args)
    classes = synth.SHAPE_CLASSES if args.classes == "shape" else synth.POSITION_CLASSES
    frames, labels, infos = synth.synthesize_dataset(
        reference, classes=classes, n_per_class=args.n, seed=args.seed
    )
    detector = R.TouchDetector(reference, threshold=args.threshold, min_area=args.min_area)

    # touch detection
    tp = fp = tn = fn = 0
    loc_errors = []
    cell_hits = cell_total = 0
    for frame, label, info in zip(frames, labels, infos):
        result = detector.detect(frame)
        if label == "none":
            if result.touch:
                fp += 1
            else:
                tn += 1
            continue
        if result.touch:
            tp += 1
            truth = synth.true_centroid(info)
            if truth is not None and result.centroid is not None:
                loc_errors.append(float(np.hypot(result.centroid[0] - truth[0], result.centroid[1] - truth[1])))
            truth_cell = None
            if truth is not None:
                h, w = reference.shape[:2]
                truth_cell = P.grid_cell((truth[0] / w, truth[1] / h))
            if truth_cell is not None and result.grid_cell is not None:
                cell_total += 1
                cell_hits += int(truth_cell == result.grid_cell)
        else:
            fn += 1

    extractor = R.FeatureExtractor(reference, threshold=args.threshold, min_area=args.min_area)
    X = np.stack([extractor.extract(f) for f in frames])
    results = {}
    for method in args.methods:
        if len(set(labels)) < 2:
            continue
        results[method] = R.evaluate(X, labels, method=method, test_size=args.test_size, seed=args.seed)

    payload: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_source": source,
        "reference_shape": list(reference.shape),
        "classes": args.classes,
        "n_per_class": args.n,
        "threshold": args.threshold,
        "min_area": args.min_area,
        "touch_detection": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": round(tp / max(1, tp + fn), 4),
            "fpr": round(fp / max(1, fp + tn), 4),
            "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        },
        "localization": {
            "n": len(loc_errors),
            "mean_error_px": round(float(np.mean(loc_errors)), 2) if loc_errors else None,
            "median_error_px": round(float(np.median(loc_errors)), 2) if loc_errors else None,
            "p90_error_px": round(float(np.percentile(loc_errors, 90)), 2) if loc_errors else None,
            "cell_accuracy": round(cell_hits / max(1, cell_total), 4),
        },
        "classification": {
            m: {
                "accuracy": round(float(r["accuracy"]), 4),
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "labels": r["labels"],
                "confusion": r["confusion"],
            }
            for m, r in results.items()
        },
    }
    _log(json.dumps(payload, indent=2))
    if args.out:
        _save_json(args.out, payload)
        _log(f"saved {args.out}")
    return 0


def cmd_collect(args) -> int:
    """Interactively collect real labelled samples (for the learned classifier)."""
    reference = _reference_for(args)
    camera = _camera_from_args(args)
    extractor = R.FeatureExtractor(reference, threshold=args.threshold, min_area=args.min_area)
    dataset_path = args.dataset
    if os.path.isfile(dataset_path):
        dataset = R.FeatureDataset.load(dataset_path)
    else:
        dataset = R.FeatureDataset(np.zeros((0, extractor.dim), np.float32), [])
    try:
        for label in args.labels:
            input(f"Press the object for label {label!r} on the gel, then press Enter ...")
            for _ in range(int(args.frames)):
                frame = camera.read()
                dataset.add(extractor.extract(frame), label)
            _log(f"collected {args.frames} frames for {label!r} (total {len(dataset)})")
    finally:
        camera.close()
    dataset.save(dataset_path)
    _log(f"dataset saved: {dataset_path}  summary={dataset.summary()}")
    return 0


def cmd_synth(args) -> int:
    reference = synth.synthetic_reference() if args.synthetic_reference else load_reference(args.reference)
    classes = synth.SHAPE_CLASSES if args.classes == "shape" else synth.POSITION_CLASSES
    frames, labels, infos = synth.synthesize_dataset(reference, classes=classes, n_per_class=args.n, seed=args.seed)
    extractor = R.FeatureExtractor(reference, threshold=args.threshold, min_area=args.min_area)
    X = np.stack([extractor.extract(f) for f in frames])
    dataset = R.FeatureDataset(X, labels)
    dataset.save(args.out)
    _log(f"saved synthetic dataset {args.out}: {dataset.summary()}")
    return 0


def cmd_train(args) -> int:
    dataset = R.FeatureDataset.load(args.dataset)
    _log(f"dataset {args.dataset}: {len(dataset)} samples, classes={dataset.summary()}")
    report = R.evaluate(
        dataset.X, dataset.y, method=args.method, test_size=args.test_size, seed=args.seed
    )
    clf = report["classifier"]
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "accuracy": report["accuracy"],
        "labels": report["labels"],
        "confusion": report["confusion"],
    }
    _log(json.dumps(payload, indent=2))
    if args.model:
        clf.fit(dataset.X, dataset.y)
        clf.save(args.model)
        _log(f"model saved: {args.model}")
    if args.out:
        _save_json(args.out, payload)
    return 0


def cmd_predict(args) -> int:
    reference = _reference_for(args)
    clf = R.DigitClassifier.load(args.model)
    extractor = R.FeatureExtractor(reference, threshold=args.threshold, min_area=args.min_area)
    camera = _camera_from_args(args)
    try:
        _log(f"live prediction with {args.model} (classes={clf.classes_}) - Ctrl-C to stop")
        for i in range(int(args.frames)):
            label, conf = clf.predict_one(extractor.extract(camera.read()))
            _log(f"{i:5d}  {label:14s} conf={conf:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
    return 0


def cmd_reset(args) -> int:
    from .usb_reset import reset

    try:
        node = reset(serial=args.serial, wait=args.wait)
    except PermissionError:
        _log("permission denied on the usbfs node: install the udev rule (make install-udev) or run with sudo")
        return 1
    except Exception as exc:  # noqa: BLE001
        _log(f"reset failed: {exc}")
        return 1
    _log(f"reset {node} OK - wait a few seconds")
    return 0


def cmd_report(args) -> int:
    """Print a compact machine-readable snapshot for the report/tests."""
    try:
        dev = find_digit(args.serial)
    except RuntimeError as exc:
        _log(json.dumps({"error": str(exc)}))
        return 1
    payload = {
        "version": __version__,
        "serial": dev.serial,
        "node": dev.dev_name,
        "by_id": dev.by_id,
        "modes": supported_modes(dev.dev_name),
    }
    _log(json.dumps(payload, indent=2))
    if args.out:
        _save_json(args.out, payload)
    return 0


def cmd_demo(args) -> int:
    """Sensor-free visual demo: render every processing view to PNG files.

    Uses a real reference if ``--reference`` is given, otherwise a synthetic
    one, and animates a synthetic contact.  Output is for documentation and for
    machines where the sensor is unavailable; it is clearly not live data.
    """
    from .fake import animated_sequence

    reference = load_reference(args.reference) if args.reference else synth.synthetic_reference()
    outdir = args.out or os.path.join("docs", "images")
    os.makedirs(outdir, exist_ok=True)

    frames = animated_sequence(reference, n=int(args.frames), fps=int(args.fps), seed=int(args.seed))
    processor = P.TactileProcessor(reference, threshold=args.threshold, min_area=args.min_area, depth=True)
    results = [processor.process(f) for f in frames]
    best = max(range(len(results)), key=lambda i: results[i].stats.force)
    tactile = results[best]
    # flow over a few frames, so the contact displacement is clearly visible
    flow_a = max(0, best - 6)
    flow = flow_mod.dense_flow(P.to_gray(frames[flow_a]), P.to_gray(frames[best]))
    shear = float(flow_mod.shear_from_dense(flow, tactile.mask).get("magnitude", 0.0))
    slip_detector = flow_mod.SlipDetector()
    slip_event = None
    for i in range(1, best + 1):
        fl = flow_mod.dense_flow(P.to_gray(frames[i - 1]), P.to_gray(frames[i]))
        shear_i = float(flow_mod.shear_from_dense(fl, results[i].mask).get("magnitude", 0.0))
        slip_event = slip_detector.update(results[i].stats.centroid, shear_i)

    saved = {
        "reference": reference,
        "view_camera": viz.single_panel(tactile, "camera"),
        "view_diff": viz.single_panel(tactile, "diff"),
        "view_contact": viz.single_panel(tactile, "contact"),
        "view_depth": viz.single_panel(tactile, "depth"),
        "view_normal": viz.single_panel(tactile, "normal"),
        "view_flow": P.label_above(viz.fit_tile(flow_mod.flow_vis(flow)[0]), "optical flow (hue=direction)"),
    }
    # sparse LK corners + arrows: more readable than dense flow on a low-texture gel
    p0, p1, status = flow_mod.sparse_flow(P.to_gray(frames[flow_a]), P.to_gray(frames[best]))
    arrows = tactile.frame.copy()
    n_good = 0
    if p0 is not None and len(p0):
        good = status.reshape(-1) == 1
        for (x0, y0), (x1, y1) in zip(p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]):
            cv2.arrowedLine(
                arrows,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                (0, 255, 255),
                1,
                tipLength=0.3,
            )
            n_good += 1
    cv2.putText(arrows, f"LK corners n={n_good}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    saved["view_flow_sparse"] = P.label_above(viz.fit_tile(arrows), "sparse LK flow (arrows)")
    saved["live_dashboard"] = viz.dashboard(
        tactile,
        flow=flow,
        slip_text=f"slip={bool(slip_event.slip) if slip_event else False} shear={shear:.2f}px",
    )
    # slip montage: four contact panels following the moving contact
    montage = []
    for offset in (0, 3, 6, 9):
        idx = min(len(results) - 1, best + offset)
        panel = P.draw_contact(P.overlay_mask(results[idx].frame, results[idx].mask), results[idx].stats)
        montage.append(P.label_above(viz.fit_tile(panel), f"frame {idx}"))
    saved["slip_sequence"] = np.hstack(montage)

    paths = {}
    for name, image in saved.items():
        path = os.path.join(outdir, f"{name}.png")
        cv2.imwrite(path, image)
        paths[name] = path

    payload = {
        "source": "synthetic animated contact" + (" on real reference" if args.reference else ""),
        "reference": args.reference,
        "frames": len(frames),
        "best_frame": int(best),
        "best_stats": tactile.stats.as_dict(),
        "best_cell": tactile.grid_cell,
        "slip": slip_event.as_dict() if slip_event else None,
        "images": paths,
    }
    _log(json.dumps(payload, indent=2))
    return 0


def cmd_selftest(args) -> int:
    """Sensor-free smoke test of the whole processing/recognition stack."""
    import tempfile

    reference = synth.synthetic_reference()
    frames, labels, infos = synth.synthesize_dataset(reference, n_per_class=4, seed=1)
    detector = R.TouchDetector(reference)
    detected = sum(int(detector.detect(f).touch) for f in frames)
    extractor = R.FeatureExtractor(reference)
    X = np.stack([extractor.extract(f) for f in frames])
    report = R.evaluate(X, labels, method="knn")
    with tempfile.TemporaryDirectory() as tmp:
        from .recording import SessionWriter, iter_session, session_meta

        path = os.path.join(tmp, "s")
        with SessionWriter(path, width=reference.shape[1], height=reference.shape[0], fps=30) as writer:
            for f in frames[:3]:
                writer.add(f)
        loaded = list(iter_session(path))
        meta = session_meta(path)
    _log(f"selftest: reference={reference.shape}, samples={len(frames)}, detected={detected}/{len(frames)}")
    _log(f"selftest: kNN accuracy={report['accuracy']:.3f} on {'/'.join(report['labels'])}")
    _log(f"selftest: recording round-trip {meta.n_frames} frames, loaded {len(loaded)}")
    return 0


# ---------------------------------------------------------------------------
# parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digit",
        description="DIGIT tactile sensor toolkit (capture, view, process, recognise, record).",
    )
    parser.add_argument("--version", action="version", version=f"digit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--serial", default=None, help="sensor serial (default: first DIGIT)")
        p.add_argument("--node", default=None, help="force a /dev/videoX node")
        p.add_argument("--resolution", default=None, help="WxH, e.g. 640x480")
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--led", type=int, default=15, help="LED level 0..15")
        p.add_argument("--raw", action="store_true", help="sensor-native orientation")

    p = sub.add_parser("list", help="list DIGIT video nodes")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("devices", help="alias for list")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="serial, node and supported modes")
    add_common(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("modes", help="list supported V4L2 modes")
    p.add_argument("--node", default=None)
    p.set_defaults(func=cmd_modes)

    p = sub.add_parser("check", help="sensor found? frames arriving? fps?")
    add_common(p)
    p.add_argument("--frames", type=int, default=45)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("snapshot", help="save one frame")
    add_common(p)
    p.add_argument("--out", default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("reference", help="capture a no-contact reference")
    add_common(p)
    p.add_argument("--out", default="reference.png")
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("led", help="set LED intensity")
    add_common(p)
    p.add_argument("level", nargs="?", type=int, default=15)
    p.add_argument("--rgb", nargs=3, type=int, default=None, metavar=("R", "G", "B"))
    p.set_defaults(func=cmd_led)

    p = sub.add_parser("fps", help="set and verify the frame rate")
    add_common(p)
    p.add_argument("fps", type=int)
    p.set_defaults(func=cmd_fps)

    p = sub.add_parser("resolution", help="set and verify the resolution")
    add_common(p)
    p.add_argument("resolution")
    p.set_defaults(func=cmd_resolution)

    p = sub.add_parser("benchmark", help="measure fps at every supported mode")
    add_common(p)
    p.add_argument("--frames", type=int, default=45)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("view", help="live view / processing dashboard")
    add_common(p)
    p.add_argument("--mode", default="dashboard", choices=["dashboard", "camera", "diff", "contact", "depth", "normal"])
    p.add_argument("--reference", default=None, help="reference PNG/NPY")
    p.add_argument("--out", default=None, help="save composed frames to PNG or folder")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--show", action="store_true", default=None, help="force an OpenCV window")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--diff-gain", type=float, default=3.0)
    p.add_argument("--no-slip", action="store_true")
    p.add_argument("--record", default=None, help="also record raw frames to this session folder")
    p.add_argument("--video", action="store_true", help="write preview video while recording")
    p.add_argument("--snapshot-dir", default="snapshots")
    p.add_argument("--status-every", type=float, default=1.0)
    p.add_argument("--fake", action="store_true", help="synthetic source, no sensor")
    p.add_argument("--fake-reference", default=None, help="reference used by --fake")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("record", help="record a session to a folder")
    add_common(p)
    p.add_argument("--out", default=None, help="session folder")
    p.add_argument("--root", default="sessions")
    p.add_argument("--name", default=None)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--reference", default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--fake", action="store_true", help="synthetic source, no sensor")
    p.add_argument("--fake-reference", default=None)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("sessions", help="list recorded sessions")
    p.add_argument("--root", default="sessions")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("play", help="replay a recorded session")
    p.add_argument("session", nargs="?", default="sessions/latest")
    p.add_argument("--root", default="sessions")
    p.add_argument("--out", default=None, help="output .avi or folder of PNGs")
    p.add_argument("--out-json", default=None)
    p.add_argument("--reference", default=None)
    p.add_argument("--process", default="none", choices=["none", "dashboard", "contact", "diff", "depth"])
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("monitor", help="headless touch/slip monitor (JSON lines with --json)")
    add_common(p)
    p.add_argument("--reference", default=None)
    p.add_argument("--ref-frames", type=int, default=15)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--json", action="store_true")
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--fake", action="store_true", help="synthetic source, no sensor")
    p.add_argument("--fake-reference", default=None)
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("eval", help="synthetic recognition evaluation (works without a sensor)")
    add_common(p)
    p.add_argument("--reference", default=None)
    p.add_argument("--synthetic-reference", action="store_true")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--ref-frames", type=int, default=15)
    p.add_argument("--classes", default="position", choices=["position", "shape"])
    p.add_argument("--n", type=int, default=10, help="samples per class")
    p.add_argument("--methods", nargs="+", default=["knn", "svm", "logreg"])
    p.add_argument("--test-size", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--out", default=None, help="save the JSON result")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("collect", help="interactively collect labelled real samples")
    add_common(p)
    p.add_argument("labels", nargs="+")
    p.add_argument("--reference", default=None)
    p.add_argument("--synthetic-reference", action="store_true")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--ref-frames", type=int, default=15)
    p.add_argument("--dataset", default="datasets/samples.npz")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("synth", help="build a synthetic feature dataset from a reference")
    p.add_argument("--reference", default=None)
    p.add_argument("--synthetic-reference", action="store_true")
    p.add_argument("--classes", default="position", choices=["position", "shape"])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="datasets/synthetic.npz")
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train", help="train/evaluate a classifier on a feature dataset")
    p.add_argument("dataset")
    p.add_argument("--method", default="svm", choices=["knn", "svm", "logreg"])
    p.add_argument("--test-size", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None, help="save the fitted model here")
    p.add_argument("--out", default=None, help="save metrics JSON here")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="live prediction with a trained model")
    add_common(p)
    p.add_argument("model")
    p.add_argument("--reference", default=None)
    p.add_argument("--ref-frames", type=int, default=15)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--fake", action="store_true", help="synthetic source, no sensor")
    p.add_argument("--fake-reference", default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("reset", help="soft USB reset (like a replug)")
    p.add_argument("--serial", default=None)
    p.add_argument("--wait", type=float, default=2.0)
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("report", help="JSON snapshot of device + modes")
    p.add_argument("--serial", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo", help="sensor-free visual demo: render every view to PNG")
    p.add_argument("--reference", default=None, help="real reference PNG/NPY (optional)")
    p.add_argument("--out", default=os.path.join("docs", "images"))
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--min-area", type=int, default=40)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("selftest", help="sensor-free smoke test")
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _log("interrupted")
        return 130
    except CameraError as exc:
        _log(f"camera error: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
