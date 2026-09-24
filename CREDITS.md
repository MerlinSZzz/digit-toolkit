# Credits and licences

This toolkit was written for the lab; it follows the shape of the existing
GelSight toolkit (`yumi-mobile/deploy/thor/gelsight`) and draws on the
open-source DIGIT ecosystem. **No third-party source code is vendored in this
folder**; the projects below were used as references, as dependencies, or as
inspiration. Licences were checked from the installed package metadata and from
the repositories' `LICENSE` files on 2026-09-24.

## DIGIT / tactile projects

| project | licence | how it is used here |
| --- | --- | --- |
| [facebookresearch/digit-interface](https://github.com/facebookresearch/digit-interface) | **CC-BY-NC-4.0** | Reference for the capture conventions only: the official image orientation (`transpose` + vertical flip), the LED packing `(r<<8)|(g<<4)|b` written to the UVC "Zoom, Absolute" control, and the nominal mode table. **Not imported.** It is re-implemented (`digit/capture.py`) because its `connect()` hard-codes QVGA@60 and its device list can pick the metadata node. |
| [facebookresearch/PyTouch](https://github.com/facebookresearch/PyTouch) | **MIT** | Inspiration for the recognition feature set (touch / no-touch, contact localisation, slip). No PyTouch code is used; the learned classifier is a small scikit-learn wrapper. |
| [facebookresearch/digit-design](https://github.com/facebookresearch/digit-design) | **CC-BY-NC-4.0** | Hardware/design reference (what the DIGIT is, the RGB LED layout). No files or code used. |
| [gelsightinc/gsrobotics](https://github.com/gelsightinc/gsrobotics) | **GPL-3.0** | Inspiration for the difference-image / contact-area / photometric-normal / Poisson-integration family of features. **Not vendored**; the Poisson/DCT solver in `digit/processing.py` is an independent implementation of the standard algorithm. |
| lab GelSight toolkit `deploy/thor/gelsight` | internal | Direct model for the *shape* of the deliverable: one Makefile of commands, a live viewer, a headless monitor, a udev rule and a USB reset. Read only; not modified. |

## Runtime dependencies

Installed by `make setup` (`requirements.txt`); all are imported normally.

| package | licence | used for |
| --- | --- | --- |
| [numpy](https://numpy.org/) | BSD-3-Clause | arrays, math |
| [opencv-python](https://github.com/opencv/opencv-python) | Apache-2.0 | capture, image processing, optical flow, drawing |
| [scipy](https://scipy.org/) | BSD-3-Clause | DCT Poisson integration |
| [scikit-learn](https://scikit-learn.org/) | BSD-3-Clause | kNN / SVM / logistic-regression classifier |
| [linuxpy](https://github.com/tiagocoutinho/linuxpy) | **GPL-3.0-or-later** | exact V4L2 mode/format enumeration (`supported_modes`); optional — `digit/device.py` falls back to the documented mode table if it is absent |
| [pyudev](https://github.com/pyudev/pyudev) | LGPL-2.1+ | device discovery (serial, capability, `by-id` path) |
| [pytest](https://pytest.org/) | MIT | the sensor-free test suite |

### Note on copyleft

`linuxpy` (GPL-3.0-or-later) and `pyudev` (LGPL-2.1+) are imported. If you plan
to redistribute this toolkit, review those licences first. For pure internal lab
use none of this is a problem. The toolkit degrades gracefully without
`linuxpy` (mode enumeration falls back to a static table) and without `pyudev`
(device discovery falls back to a `/sys` scan), so those imports can be removed
if a permissive-only distribution is needed.

The CC-BY-NC-4.0 projects (`digit-interface`, `digit-design`) are **not**
vendored and are not required; they are credited as references and may not be
used commercially without permission.
