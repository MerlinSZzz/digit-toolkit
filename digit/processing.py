"""Classical tactile processing: reference, difference, contact, depth, normals.

Everything here is pure NumPy/OpenCV and testable without a sensor.  All the
"depth"/"force" quantities are **relative and uncalibrated** -- they are
monotonic proxies, not millimetres or newtons (see FEATURES.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# basic colour helpers


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def as_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# reference / difference


def difference(
    frame: np.ndarray,
    reference: np.ndarray,
    gain: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed difference ``frame - reference``.

    Returns ``(visual, signed)`` where ``visual`` is a BGR uint8 image with the
    signed difference amplified by ``gain`` around mid-grey (128 = no change),
    and ``signed`` is the raw float difference with the frame's shape.
    """
    frame = as_float(frame)
    ref = as_float(reference)
    if frame.shape != ref.shape:
        raise ValueError(f"frame {frame.shape} and reference {ref.shape} differ")
    signed = frame - ref
    visual = np.clip(128.0 + gain * signed, 0, 255).astype(np.uint8)
    return visual, signed


def magnitude(signed: np.ndarray) -> np.ndarray:
    """Per-pixel change magnitude: max ``|frame - reference|`` over channels."""
    if signed.ndim == 2:
        return np.abs(signed).astype(np.float32)
    return np.max(np.abs(signed), axis=2).astype(np.float32)


def active_region(
    reference: np.ndarray,
    percentile: float = 15.0,
    blur: int = 5,
    min_frac: float = 0.05,
) -> np.ndarray:
    """Auto-detect the usable gel area from a no-contact reference frame.

    The DIGIT images have a dark sensor-housing border and dust/dead-pixel
    specks.  We threshold the reference brightness above a low percentile,
    keep the largest connected component, and fill its convex hull, so border
    darkness never registers as contact.
    """
    gray = to_gray(reference)
    if blur and blur > 1:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    thr = float(np.percentile(gray, percentile))
    mask = (gray > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.ones(gray.shape, np.uint8) * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        region = np.zeros_like(region)
        cv2.drawContours(region, [hull], -1, 255, thickness=cv2.FILLED)
    # never return an empty / implausibly small mask
    if region.sum() == 0 or (float((region > 0).mean()) < min_frac):
        return np.ones(gray.shape, np.uint8) * 255
    return region


def region_bbox(region: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box ``(x, y, w, h)`` of nonzero pixels, or ``None``."""
    ys, xs = np.nonzero(region)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


# ---------------------------------------------------------------------------
# contact detection


@dataclass
class ContactStats:
    """Geometry and magnitude of one contact blob."""

    area_px: int = 0
    area_frac: float = 0.0
    centroid: Optional[Tuple[float, float]] = None  # (x, y) in pixels
    bbox: Optional[Tuple[int, int, int, int]] = None
    mean_mag: float = 0.0
    peak_mag: float = 0.0
    force: float = 0.0
    n_blobs: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "area_px": self.area_px,
            "area_frac": round(float(self.area_frac), 4),
            "centroid": None
            if self.centroid is None
            else [round(float(self.centroid[0]), 1), round(float(self.centroid[1]), 1)],
            "bbox": None if self.bbox is None else [int(v) for v in self.bbox],
            "mean_mag": round(float(self.mean_mag), 2),
            "peak_mag": round(float(self.peak_mag), 2),
            "force": round(float(self.force), 1),
            "n_blobs": int(self.n_blobs),
        }

    @property
    def touch(self) -> bool:
        return self.area_px > 0


def contact_mask(
    signed: np.ndarray,
    threshold: float = 10.0,
    min_area: int = 40,
    region: Optional[np.ndarray] = None,
    open_ksize: int = 3,
    close_ksize: int = 7,
) -> np.ndarray:
    """Binary uint8 mask of pixels that changed by more than ``threshold``.

    ``region`` (from :func:`active_region`) restricts detection to the gel.
    Small components below ``min_area`` are removed.  ``close_ksize`` bridges
    the sparse texture of a light touch; use 0 to disable.
    """
    mag = magnitude(signed)
    mask = (mag > float(threshold)).astype(np.uint8) * 255
    if region is not None:
        mask = cv2.bitwise_and(mask, region)
    if open_ksize and open_ksize > 1:
        k = open_ksize if open_ksize % 2 == 1 else open_ksize + 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if close_ksize and close_ksize > 1:
        k = close_ksize if close_ksize % 2 == 1 else close_ksize + 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if min_area and min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
                keep[labels == i] = 255
        mask = keep
    return mask


def contact_stats(
    mask: np.ndarray,
    signed: Optional[np.ndarray] = None,
    region: Optional[np.ndarray] = None,
) -> ContactStats:
    """Compute area, centroid, bbox and magnitude statistics for a mask."""
    stats = ContactStats()
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return stats
    stats.area_px = int(xs.size)
    denom = float((region > 0).sum()) if region is not None else float(mask.size)
    stats.area_frac = float(xs.size) / denom if denom else 0.0
    stats.centroid = (float(xs.mean()), float(ys.mean()))
    stats.bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    m = None if signed is None else magnitude(signed)
    if m is not None:
        vals = m[mask > 0]
        stats.mean_mag = float(vals.mean())
        stats.peak_mag = float(vals.max())
        stats.force = float(vals.sum())
    n, _ = cv2.connectedComponents(mask, 8)
    stats.n_blobs = max(0, n - 1)
    return stats


def force_proxy(signed: np.ndarray, mask: np.ndarray) -> float:
    """Sum of change magnitude over the contact mask (arbitrary units).

    Grows roughly with contact area x pressure, so it is a useful **relative**
    normal-force trend, but it is not calibrated to newtons.
    """
    m = magnitude(signed)
    return float(m[mask > 0].sum()) if mask.any() else 0.0


def localize(stats: ContactStats, shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """Centroid as normalized ``(x/W, y/H)`` in ``0..1``."""
    if stats.centroid is None:
        return None
    height, width = shape[:2]
    return (stats.centroid[0] / width, stats.centroid[1] / height)


def grid_cell(normalized_xy: Optional[Tuple[float, float]], rows: int = 3, cols: int = 3) -> Optional[str]:
    """Map a normalized centroid to a ``rows x cols`` cell name, e.g. ``"top-left"``."""
    if normalized_xy is None:
        return None
    x = min(max(normalized_xy[0], 0.0), 0.9999)
    y = min(max(normalized_xy[1], 0.0), 0.9999)
    col = int(x * cols)
    row = int(y * rows)
    vertical = ["top", "middle", "bottom"] if rows == 3 else [f"r{row}"]
    horizontal = ["left", "center", "right"] if cols == 3 else [f"c{col}"]
    return f"{vertical[min(row, len(vertical)-1)]}-{horizontal[min(col, len(horizontal)-1)]}"


# ---------------------------------------------------------------------------
# relative depth / normal reconstruction (GelSight-style, uncalibrated)


def poisson_dct_neumann(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Integrate a gradient field with Neumann boundaries via a DCT solver.

    Solves ``u_xx + u_yy = div(g)`` on a rectangular grid in O(N log N).
    ``gx``/``gy`` are the x/y gradients (columns/rows).  The solution is unique
    only up to an additive constant, so the mean is removed.
    """
    try:
        from scipy.fft import dctn, idctn
    except Exception as exc:  # pragma: no cover - scipy is a hard dependency here
        raise RuntimeError("scipy is required for Poisson depth integration") from exc

    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    height, width = gx.shape

    # divergence (negative adjoint of the forward differences)
    div = np.zeros((height, width), dtype=np.float64)
    div[:, :-1] += gx[:, :-1] - gx[:, 1:]
    div[:-1, :] += gy[:-1, :] - gy[1:, :]

    f_hat = dctn(div, type=2, norm="ortho")
    ky = np.arange(height)[:, None]
    kx = np.arange(width)[None, :]
    lam = 2.0 * (np.cos(np.pi * ky / height) + np.cos(np.pi * kx / width)) - 4.0
    lam[0, 0] = 1.0
    u_hat = -f_hat / lam
    u_hat[0, 0] = 0.0
    depth = idctn(u_hat, type=2, norm="ortho")
    return depth - depth.mean()


def relative_height(
    frame: np.ndarray,
    reference: np.ndarray,
    smooth: int = 5,
) -> np.ndarray:
    """Smoothed grayscale difference: a relative indentation proxy."""
    d = to_gray(frame).astype(np.float32) - to_gray(reference).astype(np.float32)
    if smooth and smooth > 1:
        k = smooth if smooth % 2 == 1 else smooth + 1
        d = cv2.GaussianBlur(d, (k, k), 0)
    return d


def normals_from_height(height: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Pseudo surface normals (H, W, 3) from a height proxy.

    ``n = normalize(-scale * dh/dx, -scale * dh/dy, 1)``.  Without a calibrated
    light model these are *relative* normals, not metric surface normals.
    """
    h = height.astype(np.float32)
    dhdx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dhdy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nz = np.ones_like(h)
    nx = -scale * dhdx
    ny = -scale * dhdy
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return np.dstack([nx / norm, ny / norm, nz / norm]).astype(np.float32)


def depth_from_normals(normals: np.ndarray) -> np.ndarray:
    """Integrate pseudo-normals into a relative depth map via Poisson."""
    nx = normals[..., 0]
    ny = normals[..., 1]
    nz = np.clip(normals[..., 2], 1e-3, None)
    gx = -nx / nz
    gy = -ny / nz
    return poisson_dct_neumann(gx, gy)


def depth_proxy(
    frame: np.ndarray,
    reference: np.ndarray,
    smooth: int = 5,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full relative pipeline: frame/ref -> height proxy -> normals -> depth.

    Returns ``(depth, normals, height)``.  Units are arbitrary; use the shape
    and sign, not the magnitude, for interpretation.
    """
    height = relative_height(frame, reference, smooth=smooth)
    normals = normals_from_height(height, scale=scale)
    depth = depth_from_normals(normals)
    return depth, normals, height


def normal_vis(normals: np.ndarray) -> np.ndarray:
    """Map normals to a BGR uint8 image (from x,y,z in -1..1)."""
    rgb = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalize01(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Min-max normalise an array to 0..1, optionally restricted to ``mask``."""
    a = arr.astype(np.float32)
    if mask is not None and mask.any():
        vals = a[mask > 0]
    else:
        vals = a.reshape(-1)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def colorize_depth(depth: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Colourmap a relative depth map to BGR uint8 (per-frame normalised)."""
    norm = normalize01(depth)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cmap)


# ---------------------------------------------------------------------------
# drawing / annotation


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    out = frame.copy()
    if not mask.any():
        return out
    layer = np.zeros_like(out)
    layer[mask > 0] = color
    return cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)


def draw_contact(
    frame: np.ndarray,
    stats: ContactStats,
    label_prefix: str = "contact",
) -> np.ndarray:
    out = frame.copy()
    if stats.bbox is None:
        return out
    x, y, w, h = stats.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if stats.centroid is not None:
        cx, cy = int(round(stats.centroid[0])), int(round(stats.centroid[1]))
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    text = f"{label_prefix} {stats.area_px}px {stats.area_frac*100:.1f}% f={stats.force:.0f}"
    cv2.putText(out, text, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def label_above(image: np.ndarray, label: str, height: int = 22) -> np.ndarray:
    """Stack a text label above an image (used by the dashboard)."""
    bar = np.zeros((height, image.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (6, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


# ---------------------------------------------------------------------------
# convenience bundle used by the live viewer


@dataclass
class TactileFrame:
    """Everything computed from one frame against a reference."""

    frame: np.ndarray
    reference: np.ndarray
    signed: np.ndarray
    diff_vis: np.ndarray
    mask: np.ndarray
    stats: ContactStats
    depth: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    height: Optional[np.ndarray] = None

    @property
    def touch(self) -> bool:
        return self.stats.touch

    @property
    def normalized_centroid(self) -> Optional[Tuple[float, float]]:
        return localize(self.stats, self.frame.shape)

    @property
    def grid_cell(self) -> Optional[str]:
        return grid_cell(self.normalized_centroid)


class TactileProcessor:
    """Stateful convenience wrapper: reference + thresholds -> per-frame results."""

    def __init__(
        self,
        reference: np.ndarray,
        threshold: float = 10.0,
        min_area: int = 40,
        diff_gain: float = 3.0,
        depth: bool = True,
        smooth: int = 5,
    ) -> None:
        self.reference = reference
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.diff_gain = float(diff_gain)
        self.want_depth = bool(depth)
        self.smooth = int(smooth)
        self.region = active_region(reference)
        self.region_area = int(self.region.sum())

    def set_reference(self, reference: np.ndarray) -> None:
        self.reference = reference
        self.region = active_region(reference)
        self.region_area = int(self.region.sum())

    def process(self, frame: np.ndarray) -> TactileFrame:
        diff_vis, signed = difference(frame, self.reference, gain=self.diff_gain)
        mask = contact_mask(
            signed,
            threshold=self.threshold,
            min_area=self.min_area,
            region=self.region,
        )
        stats = contact_stats(mask, signed, region=self.region)
        depth = normals = height = None
        if self.want_depth:
            depth, normals, height = depth_proxy(frame, self.reference, smooth=self.smooth)
        return TactileFrame(
            frame=frame,
            reference=self.reference,
            signed=signed,
            diff_vis=diff_vis,
            mask=mask,
            stats=stats,
            depth=depth,
            normals=normals,
            height=height,
        )
