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
frame rate.  `orientation=` then selects the preset applied on top:

- `"official"` (or `True`) — the stored portrait view (the historical default);
- `"operator"` (default for the CLI) — the sensor **as the operator holds it**:
  rounded end up, square cable end down, left/right not mirrored.  It is the
  official portrait flipped left-right (`cv2.flip(frame, 1)`), see §2b;
- `"raw"` (or `False`) — the sensor-native landscape buffer, no transform;
- a `{"rotate": 0, "flip_x": true, "flip_y": false}` dict for anything else.

`DigitCamera(serial, node, width, height, fps, orientation, led, warmup)`.
`orientation_name()` returns the preset name recorded in a session's
`meta.json` (`frame_orientation`), so replay knows how the frames were stored.

- `open()` / `read()` / `close()` / context manager.
- `set_led(level)` and `set_led_rgb(r, g, b)` — 0..15 per channel, packed as
  `(r<<8)|(g<<4)|b` into the UVC "Zoom, Absolute" control (the official DIGIT
  convention). Channels work: on this unit `R=15,G=0,B=0` gives `R≈105, B≈11`.
- `set_mode(w, h, fps)` / `set_fps(f)` — **close and reopen** the camera
  instead of reconfiguring a live stream, because a mid-stream fps change
  wedged this unit once. The width/height/fps properties are set only between
  `open()` and the first read; `_configure` raises if the handle is already
  streaming, and `set_mode`/`set_fps` never touch the live handle.
- `measure_fps(n)` — grab `n` frames as fast as possible.
- **Roll fix** — a YUYV buffer misalignment can wrap the image sideways around
  a single sharp vertical seam. `detect_roll(frame)` finds the seam (the
  strongest column-to-column discontinuity, accepted only when it is well
  above the median and either isolated or very strong) and returns the signed
  number of columns to roll back; `fix_roll` applies it. `read()` runs this on
  every frame, so the returned image is un-wrapped. If a correction leaves a
  strong seam (a per-row misalignment, not a global roll) for
  `roll_reopen_after` frames, the device is reopened. A physical sideways move
  does not wrap, so it is left alone. `stats` reports
  `roll_frames`, `roll_corrected`, `last_roll_shift`, `last_roll_confidence`.
  `roll_fix=False` disables it.

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

## 2b. Orientation — `digit.orientation`

**Computes.** The transform from the stored camera frame to the sensor as the
operator holds it.  Jamie's answer (2026-09-25) is the ground truth: the
**rounded end is "up"** (an arrow on the back points that way) and the
**square cable end is down**.  Pressing the guided protocol at the operator's
top-left and finding the contact at the *upper right* of the official portrait
frame identifies the stored frame as **left-right mirrored**; **up is already
up** (no vertical flip, no rotation).

- `orient(frame, rotate=0, flip_x=False, flip_y=False)` — flips then rotates.
- `apply_preset(frame, "official" | "operator" | "raw" | dict)`.
- `PRESETS`, `DEFAULT = "operator"`, `resolve(spec)`.
- `session_frame_orientation(session)` reads `meta.json.frame_orientation`;
  `to_operator(frame, source)` converts a stored frame to the operator view.

**How it was confirmed on the 2026-09-25 session.**  Even the noisy D3
detector, once the frames are x-mirrored to the operator view, puts the six
presses in the correct layout and cuts the mean localisation error from
**173.5 px (identity)** to **56.2 px (flip_x)**; `flip_y` (327.7 px) and
`rot180` (247.4 px) are far worse, and only `flip_x` passes the
top-above-middle-above-bottom / left-right layout test.  See
`docs/images/real_20260925_orientation.png` (the rounded end is labelled UP and
the stored frame is shown mirrored below) and
`docs/images/real_20260925_press_montage_corrected.png` (the six cells in the
operator view).

**Measured (synthetic `test_orientation_d4.py`).** `operator` is exactly
`cv2.flip(official, 1)`; `orient` flips/rotates as specified and a missing
`meta.json` defaults to `official` (every session recorded before D4).

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
- `deformation_map(frame, reference, sigma=40, post_sigma=30)` → signed
  high-pass grayscale difference: the broad lighting term is removed with a
  large Gaussian, leaving the sharp bright/dark lobes of an indentation.
  **Lighting-invariant** localisation (`deformation_centroid(dmap, region,
  quantile=0.3)` averages the two lobes' centroids, weighted by lobe strength).
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

**Measured (real, untouched):** the 2026-09-24 recording
(`.logs/real_press_20260924`, 1567 frames / 100 s, 640x480@30 requested) is
**entirely untouched** — Jamie confirmed he did not press. Every frame is a
false positive by definition, so it measures the real false-positive rate.
Detector defaults: `threshold=10`, `min_area=40`, reference = the per-pixel
median of frames 100–160 (frame 0 is an auto-exposure/LED warm-up frame and must
not be used):

| detection variant | false-positive frames | per-frame FPR |
| --- | --- | --- |
| raw frames (no roll fix) | 1489 / 1567 | **0.950** |
| + roll fix (`detect_roll` + `np.roll`) | 838 / 1567 | **0.535** |
| + roll fix + adaptive reference (EMA α=0.02 while no contact) | 837 / 1567 | 0.534 |
| + roll fix + `min_peak=25` (contrast gate) | 172 / 1567 | **0.110** |
| + roll fix + grayscale magnitude > 15 | 132 / 1567 | 0.084 |

Per region (per-frame FPR after the roll fix): clean 30–184 → 0.574; the ±24 px
roll bands 215–432 / 1331–1566 → 0.601 / 0.394; the large-roll band 460–1300 →
0.543. Key findings:

- The **roll artifact dominates** the raw FPR (0.95); un-wrapping removes the
  geometric false positives, taking the overall rate to 0.54 while leaving the
  genuinely clean frames unchanged.
- The residual FPR is **chroma noise**, not geometry. The difference is
  `max |ΔBGR channel|`; on untouched frames the B/R channels dither by up to
  ~56 levels while luminance barely moves (grayscale `>5` is < 0.3 % of pixels
  vs ~25 % for max-channel). A grayscale magnitude, a higher threshold, or the
  `min_peak` contrast gate removes most of it (`peak_mag` is 19 at the median
  of the false blobs vs 56 at the 90th percentile).
- The **steady ~6.4 % step** described in the brief coincides *exactly* with the
  ±24 px roll bands. After un-wrapping, frames 300 and 1400 differ from the
  clean reference by ~2–3 grey levels spread uniformly over the whole gel
  (8x8 block means within ±1 level), per-channel means move by < 0.5, and the
  grayscale diff std is 1.15 — identical to a clean frame (1.15). It is the
  capture roll, **not** a press, **not** LED/auto-exposure (brightness is flat),
  and we cannot attribute it to the cable-strain deformation Jamie
  suspected. An adaptive reference does not help (0.535 → 0.534) precisely
  because there is no slow drift to track after the roll is fixed.
- For comparison, a synthetic contact on the real gel has TPR 1.00 / FPR 0.00
  (above), so the real FPR is a property of the sensor's chroma noise and the
  max-channel metric, not of the contact model.

### Measured (real, 2026-09-25 press session)

Source: `sessions/press_20260925_092909/` (Jamie's `make press-test`,
904 frames 480x640@30, 251 untouched + 6 press cells + 252 slide + 3x38
object).  The first `make eval-session` run reported `touch_ratio 1.0` for
**every** label including the 251 untouched frames (FPR 1.0), mean "contact"
area ~282,000 px (the whole frame), localisation mean 158 px and 0 slip
events.  Two independent causes:

1. **Stale reference.** The session's `reference.png` (mean brightness 103.3)
   does not match the session's own untouched frames (mean 109.8): with
   `max-channel |Δ| > 10` it differs in **99.97 %** of pixels (the brief
   measured 97.6 % on its own copy), while two untouched frames differ in only
   ~8 %.  Against that reference the median untouched `peak_mag` is 41 and the
   **minimum is 27**, so the `min_peak` gate below cannot help: contact is
   claimed on 283,321 px of an untouched frame.  `eval-session` now builds the
   reference from the session's own untouched phase -- the per-pixel median of
   up to 60 evenly spaced `none` frames -- and only falls back to
   `reference.png` when there is no untouched phase.  An explicit
   `--reference` still overrides it, and `press-test` stores the median of the
   untouched frames it just recorded.
2. **`min_peak` was applied but never recorded.**  The Makefile passes
   `--min-peak 25` (`MIN_PEAK ?= 25`), the parser default is now 25.0 as well,
   and `TouchDetector` did receive it; the old `eval_session.json` simply did
   not print it.  With the stale reference the gate passes 100 % of untouched
   frames anyway, which is why the run still said FPR 1.0.  The JSON now
   reports `reference_source` and `min_peak`.

**Held-out detection.**  Tuning on half A (alternating frames of every label;
reference built from the tuning half's untouched frames only) and reporting on
half B gives `threshold=35`, `min_area=30`, `min_peak=40`:

| label | held-out frames | touch frames | TPR/FPR |
| --- | --- | --- | --- |
| press | 143 | 143 | TPR **1.00** |
| object | 57 | 57 | TPR **1.00** |
| untouched | 125 | 0 | FPR **0.00** |
| slide | 126 | 18 | touch 0.14 |

With the shipped defaults (`THRESHOLD=10`, `MIN_AREA=40`, `MIN_PEAK=25`) the
same session gives untouched FPR **0.004** (1/251), press TPR **0.9965**,
object TPR **1.00**, slide touch 1.00, localisation mean 157.7 px -- so the
touch/no-touch fix is the reference, not the tuning.

**Held-out localisation is usable after correcting the orientation and retuning.**
Working in the operator view (x-mirrored from the stored official portrait, §2b),
the 2026-09-25 session's half-B press frames (tuned on half A) give:

- **Retuned brightness mask** (`threshold=15`, `min_area=30`, `min_peak=45`):
  overall **39.4 px** mean (median 33.9), 3x3 cell accuracy **0.986**;
- **Lighting-invariant deformation** (`deformation_map` sigma 40 / post 30,
  `deformation_centroid` quantile 0.3): overall **56.2 px** mean (median 53.0),
  cell accuracy **0.811**.

| cell | brightness mask | deformation | frames |
| --- | --- | --- | --- |
| top-left | 36.4 px | 58.0 px | 24 |
| top-center | 54.8 px | 97.3 px | 24 |
| top-right | 20.0 px | 33.4 px | 24 |
| middle-left | 17.3 px | 26.1 px | 23 |
| middle-right | 36.9 px | 73.5 px | 24 |
| bottom-center | 76.4 px | 47.4 px | 20 |
| **overall** | **39.4 px** | **56.2 px** | **139 / 143** |

Both beat D3's 176.5 px (and its x-mirrored 89.1 px) by a wide margin, so
**localisation is usable** on this session.  The brightness mask is numerically
best because the finger's shadow is co-located with the contact and a low
threshold averages over it; the deformation method is the one that does **not
follow the lighting** (`deformation_map` removes the broad smooth lighting term
and keeps the sharp bright/dark dipole of the indentation) and is the more
robust choice when the illumination is uncontrolled.  D3's poor number came
mostly from a too-strict mask (`threshold=35` kept only the shadow edge) plus
the uncorrected mirror, not from an absent contact signal.  Per-frame, the
grayscale deformation signal is only a few levels, so this is a *tuned* result:
`examples/eval_session_holdout_d4.py` reports every number above.

**Images** (`docs/images/real_20260925_*`, operator view, magenta X = protocol
cell centre, green cross = retuned brightness-mask centroid, red cross =
lighting-invariant deformation centroid; one sentence each):

- `real_20260925_press_montage_corrected.png`: the six press cells side by side
  in the operator view (frames 276/324/372/418/467/515); every green centroid
  lands in its own 3x3 cell, and the top row is above the middle row, which is
  above bottom-centre.
- `real_20260925_orientation.png`: the operator view labelled "rounded end =
  UP" and "cable / square end = DOWN", above the stored official portrait with
  the same red marker mirrored -- the picture of which end is up.
- `real_20260925_slide_slip_d4.png`: the slide phase's touch-gated
  texture-motion energy with the slip threshold and the fired frames (top),
  and six slide frames with the green touch mask (bottom).

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
- `frame_change_energy(prev, curr, mask, region)` → mean `|grayΔ|` inside the
  mask, the small-motion proxy for tangential texture flow.
- `TextureSlipDetector(threshold=1.6, required=3, lag=5)` → keeps a short ring
  of frames, compares frame `i` with frame `i-lag` inside the touch mask, and
  fires only when gated on touch -- it never uses the contact centroid.

**Limits.** On a gel with little texture the sparse tracker finds few corners;
`shear.quality` reports the fraction of surviving contact corners. Dense flow
is costlier than the rest of the pipeline. Shear is in **pixels/frame**, not
newtons, and a stationary contact held against a moving object still needs the
background-subtraction assumption (a handheld sensor is fine; a robot finger
that moves between frames is better served by the GelSight monitor's tuned
version). The `slip` signal is a report, not an actuator command.

**Measured (real, untouched).** On the same all-untouched 2026-09-24 recording
(§3), with default thresholds (`shear>=0.6`, `centroid>=1.5`, `required=3`):

| variant | slip frames | slip events |
| --- | --- | --- |
| raw frames | 472 / 1567 (30 %) | 42 |
| + roll fix | 184 / 1567 (12 %) | 51 |

The roll fix removes the geometric false slips (the wrap creates huge global
flow), but the residual **chroma-noise false positives** still trigger the
shear channel: a noisy contact mask makes the inside-minus-outside median
jitter around the 0.6 px threshold for three frames. `MIN_PEAK` does not change
this because the shear signal is computed from the raw mask; tomorrow's real
slide test should gate slip on a gated contact or raise `shear_threshold`.
Treat the untouched slip rate as the false-positive floor.

**Measured (real, 2026-09-25 press session, slide).**  The old
`SlipDetector` was not specific: with the tuned mask it fired in 138/287 press
frames (28 events) but only 24/252 slide frames (3), all from **centroid
jitter** (shear stayed below 0.6 px/frame).  Dense Farneback flow is degenerate
on this smooth gel (frame-to-frame `median |flow|` is ~0.0 px), so D4 replaces
it with `frame_change_energy`: the mean `|gray(frame) - gray(frame_5)|` inside
the touch mask (a small-motion L1 proxy for tangential texture flow), gated on
the shipped `TouchDetector` (`threshold=10`, `min_area=40`, `min_peak=25`) and
fired after 3 consecutive frames above the threshold.  The detector
(`TextureSlipDetector`, `threshold=1.3`, `lag=5`, `required=3`) never uses the
centroid.  Half-B held-out numbers (`examples/eval_session_holdout_d4.py`):

| label | held-out frames | slip frames | events | stable-middle events |
| --- | --- | --- | --- | --- |
| untouched | 125 | 0 | 0 | 0 |
| press | 143 | 8 | 3 | **0** |
| slide | 126 | 100 | 4 | 4 |
| object | 57 | 7 | 1 | 0 |

So on this session slip **fires on slide and not on a still press** (0
stable-middle press events; the 3 press events are all in the first/last
frames of a phase, i.e. the finger arriving/leaving).  The slide signal is
clearly higher than the noise floor: untouched energy ~1.19, still press
~1.19, slide ~1.6-2.0 grayscale levels.  The event count is low only because
the operator slides continuously, so one "event" is a long run; the fired-
frame counts (100 vs 0) are the meaningful separation.  For comparison the
inside-mask and the whole-gel variant agree (slide 100 vs 96 fired frames).
`docs/images/real_20260925_slide_slip_d4.png` shows the signal and the mask.

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
hardware clock. The achieved rate is limited by the **synchronous PNG write +
MJPEG encode, not the sensor**: the 2026-09-24 100 s recording captured 1567
frames (15.7 fps) at a requested 30 fps, while `make fps FPS=30` (no encoding)
measured ~31 fps. Drop `--video` or record fewer frames if you need the full
sensor rate.

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
- `make press-test` → a guided **real** protocol (untouched → one finger at six
  places → slide → a small object), driven by Enter between phases. It captures
  a reference at the start and, when it records, replaces the stored
  `reference.png` with the median of the untouched frames it just took (an
  explicit `REFERENCE=` keeps the given one). It writes a normal session folder
  plus `labels.csv` (index, phase, label, cell) and `protocol.json`. Use it with
  the person in front of the sensor; `--fake --yes` is a sensor-free dry run.
- `make eval-session SESSION=...` → reads that folder and reports **real**
  per-frame TPR/FPR, localisation error against the intended cell centres, and
  slip events, as JSON in `eval_session.json`. The reference is rebuilt from the
  session's own untouched phase (per-pixel median); a `REFERENCE=`/`--reference`
  path overrides it. The JSON records `reference_source`, `threshold`,
  `min_area` and `min_peak`. `MIN_PEAK` (default 25) is the contrast gate from
  §3; set `MIN_PEAK=0` to compare with the raw threshold.
- `digit.fake.FakeContactSource` is the animated source used by
  `view/monitor/record/predict --fake`; it has the same interface as
  `DigitCamera`.

**Limits.** `eval` is a synthetic benchmark; it measures the processing and
learning code, not the physical sensor's real-object performance. `demo` frames
are explicitly synthetic.

## 9. Tests — `make test`

42 pytest tests, all sensor-free (`tests/`): difference/contact/centroid,
active region, Poisson integration, depth/normal shapes, dense and sparse flow,
shear background subtraction, slip state machine, touch detection, feature
determinism, classifier train/save/load, dataset balance, session round-trip,
CSV timestamps, video export, CLI parser and `selftest`. `tests/test_capture.py`
adds the safety tests: a fake OpenCV capture asserts that **no mode property is
set while a stream is open** (`set_mode`/`set_fps` must close-and-reopen), that
`_configure` refuses a live handle, and that `detect_roll` finds/undoes a wrap
but leaves a clean or physically-panned frame alone. `tests/test_session_eval.py`
adds three sensor-free tests: the reference is built from the session's own
untouched phase (not a stale `reference.png`), `eval-session` reports FPR 0 on
untouched frames where the stale reference gives FPR 1, and the `eval-session`
CLI defaults equal the Makefile's `THRESHOLD`/`MIN_AREA`/`MIN_PEAK`.
`tests/test_orientation_d4.py` adds eight more: the `operator` preset is
`flip_x` of `official`, the `raw`/`official`/`operator` resolution and the
missing-`frame_orientation` default, `deformation_centroid` recovering a
synthetic local dipole while ignoring a broad brightness shift, and
`TextureSlipDetector` firing only on touch **and** motion. Run `make test`.
`examples/eval_session_holdout_d4.py` is the reproducible half/half evaluation
used for the real 2026-09-25 numbers above (the older
`examples/eval_session_holdout.py` is the D3 version, kept for provenance).
