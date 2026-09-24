"""Device discovery and V4L2 capability enumeration for the DIGIT sensor.

The DIGIT exposes one UVC camera with **two** video nodes of the same name:
``video0`` is the capture node, ``video1`` is a metadata node
(``META_CAPTURE``).  Opening the wrong one fails.  Discovery here therefore
records which node is actually capture-capable and always prefers it.

The nominal USB ids are vendor ``2833`` and product ``0209`` (Meta/Oculus
"Facebook DIGIT").  The serial number is available from udev as
``ID_SERIAL_SHORT`` (for the unit in this lab: ``D20576``).
"""

from __future__ import annotations

import glob
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

DIGIT_VID = "2833"
DIGIT_PID = "0209"
DIGIT_MODEL = "DIGIT"

#: udev symlink glob used as a stable device path (survives node renumbering).
BY_ID_GLOB = "/dev/v4l/by-id/usb-*DIGIT*"

#: Set to a serial string to always pick that unit; ``None`` = first found.
DEFAULT_SERIAL: Optional[str] = None


@dataclass
class DigitInfo:
    """One DIGIT video node as seen by Linux."""

    dev_name: str
    serial: str
    manufacturer: str = ""
    model: str = DIGIT_MODEL
    revision: str = ""
    capture: bool = True
    by_id: Optional[str] = None
    bus_info: str = ""

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def _by_id_for(dev_name: str) -> Optional[str]:
    for link in glob.glob(BY_ID_GLOB):
        try:
            if os.path.realpath(link) == os.path.realpath(dev_name):
                return link
        except OSError:
            continue
    return None


def _sysfs_digits() -> List[DigitInfo]:
    """Fallback discovery that only reads ``/sys`` (no pyudev needed)."""
    found: List[DigitInfo] = []
    for video in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_file = os.path.join(video, "name")
        try:
            with open(name_file, encoding="utf-8", errors="replace") as fh:
                name = fh.read().strip()
        except OSError:
            continue
        if DIGIT_MODEL not in name:
            continue
        dev_name = "/dev/" + os.path.basename(video)
        # walk up to the USB interface to read the serial
        serial = ""
        node = os.path.realpath(video)
        for _ in range(8):
            node = os.path.dirname(node)
            serial_file = os.path.join(node, "serial")
            if os.path.exists(serial_file):
                try:
                    with open(serial_file, encoding="utf-8", errors="replace") as fh:
                        serial = fh.read().strip()
                except OSError:
                    pass
                break
        found.append(
            DigitInfo(
                dev_name=dev_name,
                serial=serial or "unknown",
                manufacturer="Facebook",
                capture=True,
                by_id=_by_id_for(dev_name),
            )
        )
    return found


def list_digits() -> List[DigitInfo]:
    """Return every DIGIT video node, capture nodes first.

    Uses ``pyudev`` (installed as a dependency of ``digit-interface``) when
    available so the serial and capabilities are exact; otherwise falls back
    to a pure ``/sys`` scan.
    """
    devices: List[DigitInfo] = []
    try:
        import pyudev  # type: ignore
    except Exception:
        return _sysfs_digits()

    try:
        context = pyudev.Context()
        for dev in context.list_devices(subsystem="video4linux", ID_MODEL=DIGIT_MODEL):
            props = dev.properties
            caps = props.get("ID_V4L_CAPABILITIES", "") or ""
            # ":capture:" marks the usable streaming node; the metadata node
            # reports ":meta_capture:" instead.
            capture = "capture" in caps and "meta_capture" not in caps
            dev_name = props.get("DEVNAME", "")
            devices.append(
                DigitInfo(
                    dev_name=dev_name,
                    serial=props.get("ID_SERIAL_SHORT", "") or "unknown",
                    manufacturer=props.get("ID_VENDOR", ""),
                    model=props.get("ID_MODEL", DIGIT_MODEL),
                    revision=props.get("ID_REVISION", ""),
                    capture=capture,
                    by_id=_by_id_for(dev_name),
                )
            )
    except Exception:
        return _sysfs_digits()

    if not devices:
        return _sysfs_digits()

    # capture nodes first, then by device name
    devices.sort(key=lambda d: (not d.capture, d.dev_name))
    return devices


def find_digit(
    serial: Optional[str] = DEFAULT_SERIAL,
    node: Optional[str] = None,
) -> DigitInfo:
    """Find a DIGIT, preferring the node that can actually stream.

    Parameters
    ----------
    serial:
        Restrict to this unit serial.  ``None`` accepts any DIGIT.
    node:
        Use this exact video node, ignoring discovery (still reports serial
        when udev knows it).

    Raises
    ------
    RuntimeError
        If no matching DIGIT (or no capture-capable node) exists.
    """
    devices = list_digits()

    if node is not None:
        for dev in devices:
            if dev.dev_name == node:
                return dev
        # The node exists but udev did not label it: build a synthetic entry.
        return DigitInfo(dev_name=node, serial=serial or "unknown", capture=True)

    if serial is not None:
        devices = [d for d in devices if d.serial == serial]

    capture = [d for d in devices if d.capture]
    if capture:
        return capture[0]
    if devices:
        raise RuntimeError(
            "found a DIGIT but no capture-capable node (only a metadata node); "
            "is the sensor busy in another program?"
        )
    raise RuntimeError(
        "no DIGIT sensor found. Check `lsusb` for 2833:0209, that the cable is "
        "connected, and that you are in the `video` group (`id`)."
    )


def supported_modes(node: Optional[str] = None) -> List[Dict[str, object]]:
    """Enumerate the sensor's real V4L2 modes via ``linuxpy``.

    Returns a list of ``{"width", "height", "fps", "pixel_format",
    "description"}`` dicts, sorted.  Falls back to the modes documented by
    ``digit-interface`` if ``linuxpy`` is unavailable or the query fails.
    """
    if node is None:
        try:
            node = find_digit().dev_name
        except RuntimeError:
            node = "/dev/video0"

    modes: List[Dict[str, object]] = []
    try:
        from linuxpy.video.device import Device  # type: ignore

        cam = Device(node)
        cam.open()
        try:
            info = cam.info
            for fmt in info.formats:
                for size in info.frame_sizes():
                    if size.pixel_format != fmt.pixel_format:
                        continue
                    for interval in info.fps_intervals(
                        fmt.pixel_format, size.info.width, size.info.height
                    ):
                        fps = int(round(float(interval.max_fps)))
                        modes.append(
                            {
                                "width": int(size.info.width),
                                "height": int(size.info.height),
                                "fps": fps,
                                "pixel_format": str(fmt.pixel_format.name),
                                "description": fmt.description,
                            }
                        )
        finally:
            cam.close()
    except Exception:
        modes = []

    if not modes:
        fallback = [(640, 480, 30), (640, 480, 15), (320, 240, 60), (320, 240, 30)]
        modes = [
            {
                "width": w,
                "height": h,
                "fps": f,
                "pixel_format": "YUYV",
                "description": "YUYV 4:2:2 (documented default)",
            }
            for (w, h, f) in fallback
        ]

    modes.sort(key=lambda m: (-int(m["width"]), -int(m["fps"])))
    # de-duplicate (linuxpy can repeat a mode once per format)
    unique: List[Dict[str, object]] = []
    seen = set()
    for m in modes:
        key = (m["width"], m["height"], m["fps"])
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def default_mode() -> Dict[str, int]:
    """The safest default mode: full resolution, 30 fps."""
    return {"width": 640, "height": 480, "fps": 30}


def usb_device_path(serial: Optional[str] = None) -> Optional[str]:
    """Return the ``/dev/bus/usb/BBB/DDD`` path for the DIGIT, if visible."""
    for product_file in glob.glob("/sys/bus/usb/devices/*/idProduct"):
        dev_dir = os.path.dirname(product_file)
        try:
            with open(product_file, encoding="utf-8") as fh:
                if fh.read().strip() != DIGIT_PID:
                    continue
            with open(os.path.join(dev_dir, "idVendor"), encoding="utf-8") as fh:
                if fh.read().strip() != DIGIT_VID:
                    continue
            with open(os.path.join(dev_dir, "busnum"), encoding="utf-8") as fh:
                bus = int(fh.read().strip())
            with open(os.path.join(dev_dir, "devnum"), encoding="utf-8") as fh:
                num = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        if serial:
            try:
                with open(os.path.join(dev_dir, "serial"), encoding="utf-8") as fh:
                    if fh.read().strip() != serial:
                        continue
            except OSError:
                continue
        return f"/dev/bus/usb/{bus:03d}/{num:03d}"
    return None


def describe(dev: DigitInfo) -> str:
    """One-line human description of a device node."""
    kind = "capture" if dev.capture else "metadata"
    return (
        f"{dev.dev_name}  serial={dev.serial}  model={dev.model}  "
        f"rev={dev.revision}  [{kind}]"
    )


def iter_capture_nodes(devices: Optional[Iterable[DigitInfo]] = None) -> List[str]:
    """Return capture-capable node paths only."""
    devices = list(devices) if devices is not None else list_digits()
    return [d.dev_name for d in devices if d.capture]
