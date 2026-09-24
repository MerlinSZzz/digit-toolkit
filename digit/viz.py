"""Dashboard composition for the live viewer and screenshots.

Pure array operations: no window calls here, so the same code produces both the
on-screen window and the saved PNG used for reports.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import flow as flow_mod
from . import processing as P

TILE = (320, 240)


def fit_tile(image: np.ndarray, tile: Tuple[int, int] = TILE) -> np.ndarray:
    """Resize + letterbox an image into a fixed tile."""
    tw, th = tile
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((th, tw, 3), np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def stack_grid(panels: Sequence[np.ndarray], cols: int = 3) -> np.ndarray:
    """Stack labeled panels into a grid (panels are already labeled)."""
    rows: List[np.ndarray] = []
    tiles = list(panels)
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def hud(image: np.ndarray, lines: Sequence[str], x: int = 6, y: int = 16) -> np.ndarray:
    out = image.copy()
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (x, y + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (x, y + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def contact_panel(tactile: P.TactileFrame, tile: Tuple[int, int] = TILE) -> np.ndarray:
    """Frame with the contact mask overlay + centroid annotation."""
    over = P.overlay_mask(tactile.frame, tactile.mask)
    over = P.draw_contact(over, tactile.stats)
    return P.label_above(fit_tile(over, tile), "contact mask + centroid")


def depth_panel(tactile: P.TactileFrame, tile: Tuple[int, int] = TILE) -> np.ndarray:
    depth = tactile.depth
    if depth is None:
        depth = P.relative_height(tactile.frame, tactile.reference)
    return P.label_above(fit_tile(P.colorize_depth(depth), tile), "relative depth (uncalibrated)")


def normal_panel(tactile: P.TactileFrame, tile: Tuple[int, int] = TILE) -> np.ndarray:
    normals = tactile.normals
    if normals is None:
        height = P.relative_height(tactile.frame, tactile.reference)
        normals = P.normals_from_height(height)
    return P.label_above(fit_tile(P.normal_vis(normals), tile), "surface normals (relative)")


def flow_panel(flow: np.ndarray, tile: Tuple[int, int] = TILE) -> np.ndarray:
    vis, _ = flow_mod.flow_vis(flow)
    return P.label_above(fit_tile(vis, tile), "optical flow (hue=direction)")


def dashboard(
    tactile: P.TactileFrame,
    flow: Optional[np.ndarray] = None,
    slip_text: str = "",
    tile: Tuple[int, int] = TILE,
) -> np.ndarray:
    """A 3x2 labeled dashboard of the main processing views."""
    camera = P.label_above(fit_tile(tactile.frame, tile), "camera")
    diff = P.label_above(fit_tile(tactile.diff_vis, tile), "difference x gain")
    contact = contact_panel(tactile, tile)
    depth = depth_panel(tactile, tile)
    normal = normal_panel(tactile, tile)
    if flow is not None:
        flowp = flow_panel(flow, tile)
    else:
        flowp = P.label_above(fit_tile(np.zeros((16, 16, 3), np.uint8), tile), "optical flow (n/a)")
    grid = stack_grid([camera, diff, contact, depth, normal, flowp], cols=3)
    stats = tactile.stats
    touch = "TOUCH" if tactile.touch else "no contact"
    lines = [
        f"{touch}  area={stats.area_px}px ({stats.area_frac*100:.1f}%)  force={stats.force:.0f}",
        f"centroid={stats.centroid}  cell={tactile.grid_cell}",
        slip_text or "",
    ]
    return hud(grid, [ln for ln in lines if ln])


def single_panel(tactile: P.TactileFrame, mode: str) -> np.ndarray:
    """One processing view, full size, with a status HUD."""
    if mode in ("camera", "raw"):
        panel = tactile.frame.copy()
    elif mode in ("diff", "difference"):
        panel = tactile.diff_vis.copy()
    elif mode in ("contact", "mask"):
        panel = P.overlay_mask(tactile.frame, tactile.mask)
        panel = P.draw_contact(panel, tactile.stats)
    elif mode in ("depth", "depth_proxy"):
        panel = P.colorize_depth(tactile.depth if tactile.depth is not None else P.relative_height(tactile.frame, tactile.reference))
    elif mode in ("normal", "normals"):
        normals = tactile.normals if tactile.normals is not None else P.normals_from_height(
            P.relative_height(tactile.frame, tactile.reference)
        )
        panel = P.normal_vis(normals)
    else:
        panel = tactile.frame.copy()
    stats = tactile.stats
    touch = "TOUCH" if tactile.touch else "no contact"
    return hud(
        panel,
        [
            f"{touch} area={stats.area_px}px ({stats.area_frac*100:.1f}%) force={stats.force:.0f}",
            f"centroid={stats.centroid} cell={tactile.grid_cell}",
        ],
    )
