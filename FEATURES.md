# Features

Every feature below says **what it computes**, its **main parameters**, and
its **limits**. Parameter names are the ones in the Python API; the CLI/Make
exposes a subset (`THRESHOLD`, `MIN_AREA`, `LED`, `RES`, `FPS`, `SECONDS`,
`FRAMES`, `MODE`, ...).

A global honesty note first:

> All quantities derived from the image are **relative and uncalibrated**.
> `force` is a sum of pixel-intensity changes (arbitrary units), `depth` is a
> relative height in arbitrary units, `area` is in pixels. Turning any of them
> into newtons or millimetres needs a separate calibration (e.g. a ball /
> known weight) that this toolkit does not install for you.

---

## 1. Device discovery — `digit.device`

**Computes.** Finds the DIGIT's capture node, serial, model, revision and
supported V4L2 modes.

- `list_digits()` — every DIGIT video node, capture nodes first. Uses `pyudev`
  (exact `ID_V4L_CAPABILITIES`, `ID_SERIAL_SHORT`) with a pure `/sys` fallback.
- `find_digit(serial=None, node=None)` — the node that can actually stream.
  The DIGIT exposes **two** nodes with the same name; only one is
  `:capture:`, the other is `:meta_capture:`. Opening the wrong one fails.
- `supported_modes(node)` — enumerates real formats/sizes/fps with `linuxpy`
  and falls back to the `digit-interface` documented table.

**Parameters.** `serial` (default: first DIGIT), `node` (force a node).

**Limits.** If a program already holds the camera, `find_digit` still finds it
but `open()` fails with a clear message. Mode enumeration needs `linuxpy`; the
toolkit ships it, but the fallback table is used if it is missing.

## 2. Capture and LED — `digit.capture.DigitCamera`

**Computes.** BGR frames in the official portrait orientation
(`transpose` + vertical flip, the `digit-interface` convention), plus the live
frame rate.

- `DigitCamera(serial, node, width, height, fps, orientation, led, warmup)`.
- `open()` / `read()` / `close()` / context manager.
- `set_led(level)` and `set_led_rgb(r, g, b)` — 0..15 per channel, packed as
  `(r<<8)|(g<<4)|b` into the UVC "Zoom, Absolute" control (the official DIGIT
  convention). Channels work: on this unit `R=15,G=0,B=0` gives `R≈105, B≈11`.
- `set_mode(w, h, fps)` / `set_fps(f)` — **close and reopen** the camera
  instead of reconfiguring a live stream, because a mid-stream fps change
  wedged this unit once.
- `measure_fps(n)` — grab `n` frames as fast as possible.

**Parameters.** Default `640x480 @ 30`, LED 15, warmup 8 frames (the first
frames after open are dark).

**Limits.** The sensor's modes are only `640x480 @ 30/15` and `320x240 @ 60/30`
(YUYV on this unit). The `CAP_PROP_ZOOM` readback is clamped to 255 on this
unit and is not a reliable LED indicator — read the frame means instead. Do not
open a second file descriptor on the device while streaming (the firmware can
throw `EPROTO -71`); always set the LED through `DigitCamera`.

`CaptureThread` wraps the camera in a background thread and reconnects after a
USB drop, publishing the newest frame to `wait_new(last_seq)` with `fps`,
`connected` and `error` properties.

## 3. Difference and contact — `digit.processing`

**Computes.**

- `difference(frame, reference, gain)` → signed `frame - reference` (float) and
  a BGR visual centred on grey 128, amplified by `gain` (default 3).
- `magnitude(signed)` → per-pixel `max |Δchannel|`.
- `active_region(reference, percentile=15, blur=5, min_frac=0.05)` → the usable
  gel mask: threshold above a low brightness percentile, largest connected
  component, filled convex hull. Excludes the dark housing and dust.
- `contact_mask(signed, threshold=10, min_area=40, region, open_ksize=3,
  close_ksize=7)` → uint8 mask of changed pixels, restricted to the gel,
  morphologically opened (despeckle) and closed (bridge a light touch), with
  components below `min_area` removed.
- `contact_stats(mask, signed, region)` → `area_px`, `area_frac` (of the gel
  area), `centroid`, `bbox`, `mean_mag`, `peak_mag`, `force`, `n_blobs`.
- `localize(stats, shape)` → centroid normalised to `0..1`.
- `grid_cell(normalized, rows=3, cols=3)` → `"top-left"`, `"middle-center"`, ...
- `force_proxy(signed, mask)` → `sum(magnitude over mask)`, arbitrary units.
- `TactileProcessor(reference, threshold, min_area, diff_gain, depth, smooth)`
  — bundles all of the above per frame into a `TactileFrame`.

**Parameters and how to tune.**

| parameter | default | effect |
| --- | --- | --- |
| `threshold` | 10 | change magnitude (0..255) that counts as contact. Lower = more sensitive, more noise. |
| `min_area` | 40 px | ignore small speckle/reflections. |
| `diff_gain` | 3 | only the visual difference image. |
| `close_ksize` | 7 | bridges sparse texture of a light touch. |
| `active_region.percentile` | 15 | lower keeps more of the gel. |

**Limits.** `area_frac`, `force` and `threshold` depend on the LED level,
exposure and the reference quality; compare runs at the same settings. A
one-pixel geometric shift of the whole sensor (bump the table) changes the
difference everywhere — re-capture the reference, or use the background
cancellation in the shear channel. `active_region` assumes the gel is the
largest bright region; if the gel is very dark, set `threshold` lower.

**Measured (synthetic contacts on the real `reference.npy`):** touch detection
TPR **1.00**, FPR **0.00**; localisation mean error **0.26 px**, 90th
percentile **0.47 px**; 3x3 cell accuracy **1.00** (see §8 for the exact test).

## 4. Relative depth and normals — `digit.processing`

**Computes.**

- `relative_height(frame, reference, smooth=5)` → smoothed grayscale difference
  (the indentation proxy).
- `normals_from_height(height, scale)` → `n = normalize(-scale·∂h/∂x,
  -scale·∂h/∂y, 1)`, a unit normal per pixel.
- `depth_from_normals(normals)` and `poisson_dct_neumann(gx, gy)` — integrate a
  gradient field with Neumann boundaries in O(N log N) using a DCT solver
  (scipy). `depth_proxy(frame, reference)` runs the whole chain:
  frame/ref → height → normals → depth.
- `normal_vis` and `colorize_depth` render them for the viewer.

**Parameters.** `smooth` (odd Gaussian kernel, default 5), `scale` (normal
slope gain).

**Limits.** There is **no calibrated photometric-stereo light model** here, so
these are *relative* normals/depth, not millimetres. They are useful for shape
and edge visualisation and for the sign of indentation, not for metrology. The
DCT integration is validated against a synthetic Gaussian (correlation > 0.9)
in `tests/test_processing.py`.

## 5. Optical flow and slip — `digit.flow`

**Computes.**

- `dense_flow(prev, curr, ...)` → Farneback flow `(H, W, 2)`.
- `sparse_flow(prev, curr, mask, ...)` → Pyramidal Lucas-Kanade displacement of
  gel-texture corners (`goodFeaturesToTrack` + `calcOpticalFlowPyrLK`).
- `shear_from_sparse(p0, p1, status, mask)` / `shear_from_dense(flow, mask)` →
  median displacement **inside the contact minus the background**, which
  cancels global camera/mount motion (the trick from the lab's GelSight
  handover monitor).
- `SlipDetector(shear_threshold=0.6, centroid_threshold=1.5, required=3)` →
  state machine combining shear and contact-centroid speed; requires
  `required` consecutive agreeing frames to report `slip=True`.

**Limits.** On a gel with little texture the sparse tracker finds few corners;
`shear.quality` reports the fraction of surviving contact corners. Dense flow
is costlier than the rest of the pipeline. Shear is in **pixels/frame**, not
newtons, and a stationary contact held against a moving object still needs the
background-subtraction assumption (a handheld sensor is fine; a robot finger
that moves between frames is better served by the GelSight monitor's tuned
version). The `slip` signal is a report, not an actuator command.

## 6. Recording and replay — `digit.recording`

**Computes.** A self-describing session folder: lossless PNG frames +
`frames.csv` (index, **timestamp_ns**, filename) + `meta.json` (serial, node,
mode, LED, start/end, count) + optional `video.avi` and `reference.png`.

- `SessionWriter`, `record(camera, ...)`, `iter_session`, `load_session`,
  `session_meta`, `list_sessions`, `export_video`.

**Parameters.** `seconds` or `frames` (both `None` = until stopped), `root`,
`name`, `save_video`, `reference`.

**Limits.** PNG at 640x480 is ~250-400 kB/frame and the default recording is
uncompressed, so a 30 fps session grows quickly; use `SECONDS=` and the AVI for
a quick look. Timestamps are host `time.time_ns()` at capture, not the camera's
hardware clock.

## 7. Recognition — `digit.recognition`

**Computes.**

- **Touch / no-touch** and **where**: `TouchDetector(reference, threshold,
  min_area)` returns `touch`, `area_px`, `centroid`, normalised centroid and a
  3x3 `cell`. This is the geometric detector from §3, live.
- **Learned classification**: `FeatureExtractor` builds a 163-dim deterministic
  vector (12x12 pooled change magnitude L2-normalised, 3 global magnitude
  stats, 3 mask-geometry values, 7 log Hu moments, frame BGR mean, in-mask
  change mean). `DigitClassifier(method="knn"|"svm"|"logreg")` wraps
  scikit-learn with feature standardisation and pickled save/load.
  `collect` gathers **real** labelled samples; `synth` builds synthetic ones;
  `train`/`evaluate` reports accuracy and a confusion matrix.

**Parameters.** `grid`, `threshold`, `min_area` for features; `method`,
`n_neighbors`/`C`/`kernel`, `test_size`, `seed` for the classifier.

**Limits and honest numbers.** The classifier is only as good as its data. The
numbers below are measured on **synthetic** indentation fields added to a real
no-contact reference, **not** on real objects — the agent building this toolkit
could not physically press the gel (see the report). Synthetic contacts are
cleaner than real ones, so treat these as an upper bound:

| task | classes | samples | method | accuracy |
| --- | --- | --- | --- | --- |
| contact position | none + 5 positions | 10/class (60) | kNN / SVM / logreg | 1.00 / 1.00 / 1.00 |
| object shape | none, blunt, sharp, edge, ring, two-finger | 15/class (90) | kNN / SVM / logreg | 0.97 / 1.00 / 1.00 |

Touch detection / localisation (same synthetic test): TPR 1.00, FPR 0.00,
centroid error 0.26 px. On **real** data, expect lower accuracy and re-measure
with `make collect` + `make train`; the code prints its own confusion matrix so
you never have to trust a claim you did not measure.

## 8. Evaluation and demo — `digit.cli`

**Computes.**

- `make eval` → touch detection (TPR/FPR), localisation (mean/median/p90 pixel
  error, 3x3 cell accuracy) and classifier accuracy, printed as JSON and saved
  to `.logs/eval.json`. `REFERENCE=reference.npy` runs the synthetic contacts
  on the real gel instead of a synthetic reference.
- `make demo` → renders every processing view from a synthetic animated contact
  (works with no sensor) to `docs/images/`.
- `digit.fake.FakeContactSource` is the animated source used by
  `view/monitor/record/predict --fake`; it has the same interface as
  `DigitCamera`.

**Limits.** `eval` is a synthetic benchmark; it measures the processing and
learning code, not the physical sensor's real-object performance. `demo` frames
are explicitly synthetic.

## 9. Tests — `make test`

33 pytest tests, all sensor-free (`tests/`): difference/contact/centroid,
active region, Poisson integration, depth/normal shapes, dense and sparse flow,
shear background subtraction, slip state machine, touch detection, feature
determinism, classifier train/save/load, dataset balance, session round-trip,
CSV timestamps, video export, CLI parser and `selftest`. Run `make test`.
