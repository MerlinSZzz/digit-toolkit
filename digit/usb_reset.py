"""Soft USB reset for the DIGIT (like the GelSight toolkit's ``usb_reset.py``).

A soft reset is the software equivalent of unplugging/replugging: it recovers a
sensor that enumerates but whose UVC stream has wedged (0-byte frames,
``EPROTO -71``).  It needs write permission on ``/dev/bus/usb/BBB/DDD``, which
the optional udev rule grants to the ``plugdev`` group; without the rule the
command fails with a clear message and you can either run it with ``sudo`` or
physically replug the sensor.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from typing import Optional

from .device import usb_device_path

USBDEVFS_RESET = 21780


def reset(serial: Optional[str] = None, wait: float = 2.0) -> str:
    """Reset the DIGIT USB device. Returns the node that was reset.

    Raises ``PermissionError`` when the usbfs node is not writable.
    """
    node = usb_device_path(serial)
    if node is None:
        raise RuntimeError(
            "DIGIT (2833:0209) not found on USB; check the cable or replug it."
        )
    fd = os.open(node, os.O_WRONLY)
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    except OSError as exc:
        # ENODEV right after the reset is normal: the device is re-enumerating.
        if exc.errno == 19:
            return node
        raise
    finally:
        os.close(fd)
    if wait:
        time.sleep(wait)
    return node


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--wait", type=float, default=2.0)
    args = parser.parse_args(argv)
    try:
        node = reset(serial=args.serial, wait=args.wait)
    except PermissionError:
        print(
            "no write permission on the usbfs node. Install the udev rule "
            "(make install-udev) once and replug, or run `sudo make reset`.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"reset failed: {exc}", file=sys.stderr)
        return 1
    print(f"reset {node} OK - wait a few seconds for re-enumeration")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
