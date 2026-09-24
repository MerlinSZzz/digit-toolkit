"""Live viewer / monitor loop for the DIGIT.

Runs the capture thread, computes the selected processing view, optionally
shows an OpenCV window and/or saves the composed frames to disk.  The saved
frames are what the reports use as screenshots when there is no display.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import cv2
import numpy as np

from . import flow as flow_mod
from . import processing as P
from . import viz
from .capture import CaptureThread, DigitCamera
from .recording import SessionWriter


def capture_reference(camera: DigitCamera, n: int = 15, verbose: bool = True) -> np.ndarray:
    """Average ``n`` frames into a clean no-contact reference.

    Keep the gel untouched while this runs (a couple of seconds).
    """
    if verbose:
        print("Capturing reference - do not touch the sensor ...", flush=True)
    frames = []
    for _ in range(max(1, int(n))):
        frames.append(camera.read())
    stack = np.stack(frames).astype(np.float32)
    ref = np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)
    if verbose:
        print(f"Reference captured from {len(frames)} frames.", flush=True)
    return ref


def load_reference(path: str) -> np.ndarray:
    """Load a reference from a ``.png/.jpg`` or a ``.npy`` file."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.uint8)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read reference image {path}")
    return img


def run_view(
    camera: DigitCamera,
    mode: str = "dashboard",
    reference: Optional[np.ndarray] = None,
    out: Optional[str] = None,
    frames: Optional[int] = None,
    seconds: Optional[float] = None,
    show: Optional[bool] = None,
    window: str = "DIGIT",
    threshold: float = 10.0,
    min_area: int = 40,
    diff_gain: float = 3.0,
    want_flow: bool = True,
    slip: bool = True,
    pause_key: bool = True,
    snapshot_dir: str = "snapshots",
    record_dir: Optional[str] = None,
    save_video: bool = False,
    status_every: float = 1.0,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the viewer. Returns a small summary dict.

    ``mode`` is one of ``dashboard``, ``camera``, ``diff``, ``contact``,
    ``depth``, ``normal``.  When ``out`` is set the composed frame is written
    to that path (a PNG or a folder).  ``show=None`` auto-detects a display.
    """
    if show is None:
        show = bool(os.environ.get("DISPLAY")) and os.environ.get("DIGIT_NO_WINDOW") != "1"

    if not camera._cap:
        camera.open()
    if reference is None:
        reference = capture_reference(camera, verbose=verbose)
    processor = P.TactileProcessor(
        reference=reference,
        threshold=threshold,
        min_area=min_area,
        diff_gain=diff_gain,
        depth=(mode in ("dashboard", "depth", "depth_proxy", "normal", "normals")),
    )
    slip_detector = flow_mod.SlipDetector() if slip else None

    thread = CaptureThread(camera)
    thread.start()

    # give the thread a moment to publish the first frame
    _, last_seq = thread.wait_new(-1, timeout=3.0)

    t_start = time.time()
    count = 0
    saved = 0
    paused = False
    prev_gray: Optional[np.ndarray] = None
    last_status = 0.0
    last_panel: Optional[np.ndarray] = None
    summary: Dict[str, object] = {"frames": 0, "saved": 0, "last_touch": False}

    os.makedirs(snapshot_dir, exist_ok=True)
    if out and not out.lower().endswith((".png", ".jpg", ".jpeg")):
        os.makedirs(out, exist_ok=True)

    writer: Optional[SessionWriter] = None
    if record_dir:
        writer = SessionWriter(
            record_dir,
            serial=camera.serial or "",
            node=camera.dev_name,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            led=camera.led,
            save_video=save_video,
            reference=reference,
        )

    try:
        while True:
            if frames is not None and count >= frames:
                break
            if seconds is not None and (time.time() - t_start) >= seconds:
                break

            frame, last_seq = thread.wait_new(last_seq, timeout=1.0)
            if frame is None:
                if not thread.connected and thread.error:
                    if verbose:
                        print(f"[viewer] {thread.error}", flush=True)
                continue

            tactile = processor.process(frame)
            if writer is not None:
                writer.add(frame)

            flow = None
            shear = 0.0
            if want_flow and prev_gray is not None:
                flow = flow_mod.dense_flow(prev_gray, P.to_gray(frame))
                if slip_detector is not None:
                    shear = float(flow_mod.shear_from_dense(flow, tactile.mask).get("magnitude", 0.0))
            prev_gray = P.to_gray(frame).copy()

            slip_text = ""
            if slip_detector is not None:
                event = slip_detector.update(tactile.stats.centroid, shear)
                slip_text = f"slip={event.slip} shear={event.magnitude:.2f}px"
                summary["last_slip"] = event.slip

            if mode == "dashboard":
                panel = viz.dashboard(tactile, flow=flow, slip_text=slip_text)
            else:
                panel = viz.single_panel(tactile, mode)
            last_panel = panel
            summary["last_touch"] = bool(tactile.touch)

            if out:
                if out.lower().endswith((".png", ".jpg", ".jpeg")):
                    if frames is not None or seconds is not None:
                        cv2.imwrite(out, panel)
                        saved += 1
                    elif count % 30 == 0:
                        cv2.imwrite(out, panel)
                        saved += 1
                else:
                    cv2.imwrite(os.path.join(out, f"{count:06d}.png"), panel)
                    saved += 1

            if show:
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    path = os.path.join(snapshot_dir, f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
                    cv2.imwrite(path, panel)
                    if verbose:
                        print(f"snapshot -> {path}", flush=True)
                if key == ord("r"):
                    reference = capture_reference(camera, verbose=verbose)
                    processor.set_reference(reference)
                    prev_gray = None
                if key == ord("p") and pause_key:
                    paused = True
                    print("paused - press any key to resume", flush=True)
                    cv2.waitKey(0)
                    paused = False

            count += 1
            now = time.time()
            if verbose and now - last_status >= status_every:
                last_status = now
                st = tactile.stats
                summary["frames"] = count
                summary["saved"] = saved
                print(
                    f"[viewer] {thread.fps:5.1f} fps  touch={tactile.touch}  "
                    f"area={st.area_px:5d}px  force={st.force:8.0f}  "
                    f"cell={tactile.grid_cell}  {slip_text}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
        thread.stop()
        thread.join(timeout=1.0)
        if show:
            cv2.destroyAllWindows()

    summary["frames"] = count
    summary["saved"] = saved
    if last_panel is not None and out and out.lower().endswith((".png", ".jpg", ".jpeg")):
        cv2.imwrite(out, last_panel)
    return summary
