# DIGIT tactile sensor toolkit

> **中文摘要**：这是给新来的 visitor 使用的 DIGIT 触觉传感器工具包。它把「读取、LED 控制、分辨率/帧率、
> 实时可视化、接触检测与定位、相对深度/法线、光流与滑动检测、录制回放、一个可训练的分类示例、测试与文档」
> 都收在一条 `make <命令>` 和 `python -m digit <命令>` 之下。五步上手：`make setup` → 插上传感器 →
> `make check` → `make view` → `make record`。所有实时量（面积、力代理、深度）都是**未经标定的相对量**，
> 不是毫米或牛顿。用法、每个功能的参数与局限见本文件与 [`FEATURES.md`](FEATURES.md)；开源项目与许可见
> [`CREDITS.md`](CREDITS.md)。

A small, beginner-friendly toolkit for the **Facebook/Meta DIGIT** optical
tactile sensor (USB `2833:0209`), modelled on the lab's GelSight toolkit
(`deploy/thor/gelsight`): one Makefile of simple commands, a live viewer, a
headless monitor, a udev rule and a USB reset.

Everything is in this folder. You do not need to know the sensor to start:
run `make setup`, plug it in, run `make check`, then `make view`.

---

## What is a DIGIT?

The DIGIT is a small, camera-based tactile sensor. A soft gel pad is lit by
RGB LEDs; a camera behind the gel photographs it. When something presses on the
gel the image deforms, and the change tells you **where** the contact is, how
**large** it is and (roughly) how **hard** the press is. It is the little
brother of the GelSight Mini: same idea, cheaper, smaller field of view, no
metric depth calibration out of the box.

On this machine the sensor is:

| fact | value |
| --- | --- |
| USB id | `2833:0209` (`lsusb` → *Oculus VR, Inc. DIGIT*) |
| serial | `D20576` |
| capture node | `/dev/video0` (the other node, `/dev/video1`, is metadata only) |
| stable path | `/dev/v4l/by-id/usb-Facebook_DIGIT_D20576-video-index0` |
| modes | `640x480 @ 30/15 fps`, `320x240 @ 60/30 fps` (YUYV) |
| LED | 12-bit RGB, 0..15 per channel, exposed through UVC "Zoom" |
| image | portrait after the official orientation fix (VGA → 480x640 shown) |

The sensor produces a colour image with a bright, slightly rainbow-lit gel and
a dark housing border. The toolkit auto-detects the usable gel area
(`processing.active_region`) so the border is never counted as contact.

---

## Five-minute quick start

```bash
cd rebecca/digit

make setup          # one time: creates the conda env 'digit' and installs deps
make check          # is it found? are frames arriving? what fps?
make reference      # average 15 untouched frames -> reference.png (touch nothing!)
make view           # live dashboard: camera | difference | contact | depth | normals | flow
```

Leave the gel untouched while `make reference` runs (~0.5 s). The reference is
the "no contact" image that every later difference is measured against.

If `make check` says *no sensor found*, see [Troubleshooting](#troubleshooting).

Interactive keys in `make view`: `q`/`Esc` quit, `s` save a screenshot to
`snapshots/`, `r` re-capture the reference, `p` pause.

---

## Commands

Run `make help` for the same list. Every command is a thin wrapper over
`python -m digit <command>`, and accepts `SERIAL=`, `NODE=`, `RES=`, `FPS=`,
`LED=`, `THRESHOLD=`, `FRAMES=` overrides, e.g. `make view LED=8 MODE=depth`.

### Setup and inspection

| command | what it does |
| --- | --- |
| `make help` | list every command (default target) |
| `make setup` | create the `digit` conda env and install `requirements.txt` |
| `make check` | sensor found? capture node? supported modes? frames arriving? measured fps? saves `.logs/check_frame.png` |
| `make info` | serial, node, revision and the supported V4L2 modes |
| `make list` | list DIGIT video nodes and their stable `by-id` paths |
| `make modes` | one line per supported resolution/fps |
| `make benchmark` | measure real fps at every supported mode, writes `.logs/benchmark.json` |
| `make udev-check` | is `99-digit.rules` installed? |
| `make reset` | soft USB reset (like a replug); needs the udev rule or sudo |
| `make install-udev` | install `99-digit.rules` (needs sudo, once) |

### Capture and control

| command | example | notes |
| --- | --- | --- |
| `make snapshot` | `make snapshot` | one frame → `snapshots/snapshot_<time>.png` |
| `make reference` | `make reference` | 15-frame average → `reference.png` and `reference.npy` |
| `make led` | `make led LEVEL=8` | set all LED channels, 0..15 |
| `make led-rgb` | `make led-rgb R=15 G=0 B=0` | independent channels (the DIGIT has RGB LEDs) |
| `make fps` | `make fps FPS=15` | open at that fps and measure it (modes: 30/15 at VGA) |
| `make resolution` | `make resolution RES=320x240` | open at that resolution and measure |

### Live views

| command | what you see |
| --- | --- |
| `make view` | 3x2 dashboard: camera, difference, contact mask, relative depth, relative normals, optical flow |
| `make view MODE=diff` | amplified signed difference (most sensitive contact view) |
| `make view MODE=contact` | frame + contact mask + centroid + area/force |
| `make view MODE=depth` | colour-mapped relative depth |
| `make view MODE=normal` | relative surface normals |
| `make save MODE=<mode> OUT=file.png FRAMES=60` | headless: render a view and write it (screenshots, no display needed) |
| `make view-diff` / `view-contact` / `view-depth` / `view-normal` | shortcuts that write `.logs/view_*.png` |
| `make monitor SECONDS=10` | headless text state: `touch`, `area`, `force`, `cell`, `slip`, `fps` |
| `make monitor-json SECONDS=10` | the same as one JSON object per frame (for a robot/front-end) |
| `make demo` | sensor-free: renders every view from a synthetic animated contact to `docs/images/` |

### Recording and replay

| command | what it does |
| --- | --- |
| `make record SECONDS=10` | record to `sessions/<timestamp>/` (PNG frames + `frames.csv` + `meta.json` + `video.avi`) |
| `make record NAME=grip SECONDS=5` | record to `sessions/grip/` |
| `make sessions` | list recorded sessions |
| `make play SESSION=sessions/grip` | print a numeric summary (frames, duration, touch ratio if a reference exists) |
| `make play SESSION=sessions/grip PROCESS=contact OUT=replay.avi` | replay as a processed video |
| `make play SESSION=sessions/grip PROCESS=dashboard OUT=sheet.png STEP=12` | contact-sheet montage |
| `make play SESSION=sessions/grip PROCESS=depth --show` (CLI only) | replay in a window |

### Recognition

| command | what it does |
| --- | --- |
| `make eval` | **measured** touch detection, localisation and classification on synthetic contacts; writes `.logs/eval.json` |
| `make eval REFERENCE=reference.npy` | the same, but synthetic contacts added to the **real** gel reference |
| `make synth` | build a synthetic feature dataset (`datasets/synthetic.npz`) |
| `make train DATASET=datasets/synthetic.npz MODEL=models/c.pkl` | train + evaluate kNN/SVM/logreg |
| `make collect LABELS="screw bolt"` | interactively collect labelled **real** samples (you press the object, Enter, then N frames are saved) |
| `make predict MODEL=models/c.pkl` | live prediction with a trained model |

### Development

| command | what it does |
| --- | --- |
| `make test` | 33 sensor-free pytest tests |
| `make selftest` | sensor-free end-to-end smoke test of processing/recognition/recording |

---

## Where files go

```
reference.png / reference.npy     no-contact reference (regenerated by `make reference`)
snapshots/                        single frames from `make snapshot` / `s` key
sessions/<name>/                  one folder per recording:
    frames/000000.png ...         lossless frames
    frames.csv                    index, timestamp_ns, filename  (per-frame timestamps)
    meta.json                     serial, node, mode, LED, start/end, count
    video.avi                     optional MJPEG preview
    reference.png                 optional reference captured with the session
datasets/                         feature datasets (.npz)
models/                           trained classifiers (.pkl)
.logs/                            check/benchmark/eval JSON and headless renders
docs/images/                      documentation and report images
```

Generated data (`sessions/`, `snapshots/`, `datasets/`, `models/`, `.logs/`,
`*.npy`) is git-ignored; the code and docs are committed.

---

## The one API you need in Python

```python
from digit import DigitCamera, find_digit
from digit import processing as P, recognition as R, recording as rec

print(find_digit().serial)                 # 'D20576'

with DigitCamera() as cam:                 # opens /dev/video0 at 640x480@30
    cam.set_led(12)
    reference = cam.read()                 # keep the gel untouched!
    frame = cam.read()
    signed = frame.astype("float32") - reference.astype("float32")
    mask = P.contact_mask(signed, threshold=10.0, min_area=40)
    stats = P.contact_stats(mask, signed)
    print(stats.centroid, stats.area_px, stats.force)

    result = R.TouchDetector(reference).detect(frame)
    print(result.as_dict())                # {'touch': ..., 'cell': ...}
```

`DigitCamera` picks the capture node automatically, applies the official
orientation, and supports per-channel LED control. See
[`FEATURES.md`](FEATURES.md) for every parameter.

---

## Troubleshooting

**`make check` says "no DIGIT sensor found".**
Check `lsusb | grep 2833` (expected `2833:0209 Oculus VR, Inc. DIGIT`). If it is
missing, it is a cable/port problem; try another port. If it is present, run
`make list`: a DIGIT always exposes two nodes and only one can stream.
`DigitCamera` prefers the capture node automatically.

**"could not open /dev/video0" / permission denied.**
Your user must be in the `video` group. Check with `id`; if not, a login-time
change is needed: `sudo usermod -aG video "$USER"` then log back in. (On this
machine `jamie` is already in `video`.)

**No frames, `select() timeout`, or a wedged sensor.**
A UVC/DIGIT can wedge after an unlucky reconfiguration or a suspend. Recover
with a soft reset if you have the udev rule: `make reset`. Otherwise replug the
sensor, or run `sudo make reset`. **Do not** change fps/resolution repeatedly
while it is streaming; the toolkit reopens the camera for a mode change to
avoid this, but other programs may not.

**Wrong node / "opened but no frames".**
Use `make list` and the stable path
`/dev/v4l/by-id/usb-Facebook_DIGIT_D20576-video-index0`, or force it:
`make check NODE=/dev/video0`. Never point at `video1`; it is metadata only.

**Washed-out or very dark image.**
Set the LED: `make led LEVEL=15` (max) or `make led LEVEL=8` (softer). The
image is auto-exposed by the camera, so raising the LED does not always raise
the mean brightness; use `make led-rgb R=15 G=0 B=0` to see the colour change.
On this unit the `CAP_PROP_ZOOM` **readback** is clamped to 255 and is not a
reliable indicator; the frame means printed by `make led` are.

**The image is mostly dark at one side.**
The camera sees the sensor housing as well as the gel. The toolkit's
`active_region` mask (default: brightness above the 15th percentile, largest
component, convex hull) removes it, and `area_frac` is measured relative to the
gel area, not the whole frame.

**Live window does not open.**
If there is no display (`echo $DISPLAY` empty, e.g. over SSH) use
`make save MODE=dashboard OUT=shot.png` or `make monitor`; both are headless.
`DIGIT_NO_WINDOW=1` forces headless mode.

**LED control seems to have no effect.**
The DIGIT maps LED intensity onto the UVC "Zoom, Absolute" control. Some other
programs/cameras also use Zoom; make sure only one program has the camera open.

---

## Credits and licence

See [`CREDITS.md`](CREDITS.md) for the full table. In short: the capture/LED
conventions follow `facebookresearch/digit-interface` (CC-BY-NC-4.0, referenced
but **not** imported); the feature set is inspired by `facebookresearch/PyTouch`
(MIT), `facebookresearch/digit-design` (CC-BY-NC-4.0), `gelsightinc/gsrobotics`
(GPL-3.0) and the lab's GelSight toolkit; the Poisson/DCT depth integration is
an independent implementation of the standard algorithm. No third-party source
is vendored. Note that the optional V4L2 enumeration dependency `linuxpy` is
GPL-3.0-or-later, so review licences before redistributing.
