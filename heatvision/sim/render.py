"""Rasterise a :class:`~heatvision.sim.scene.Scene` into an RGB + LWIR pair.

The two channels are produced by *different* physical models, which is the
whole point:

* RGB sees reflected visible light — colour, texture, specular highlights.
* LWIR sees ``e * T_object + (1 - e) * T_reflected``. Low-emissivity metal
  therefore hides a hot core, and a matte warm motor fragment shouts louder
  than a real battery.

Both frames come out pre-registered (same intrinsics, same pixel grid), which
is what a fused bispectral camera module gives you in practice.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import BeltConfig, DetectorConfig, ThermalConfig
from .scene import Item, Scene

_EPS = 1e-6


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def m_to_px(belt: BeltConfig, x_m: float, y_m: float) -> tuple[float, float]:
    """Belt metres -> image pixels (col, row)."""
    col = x_m * belt.px_per_m
    row = belt.center_row + y_m * belt.px_per_m
    return col, row


def px_to_m(belt: BeltConfig, col: float, row: float) -> tuple[float, float]:
    x_m = col / belt.px_per_m
    y_m = (row - belt.center_row) / belt.px_per_m
    return x_m, y_m


def item_polygon_px(belt: BeltConfig, item: Item) -> np.ndarray:
    pts = item.corners_m()
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] * belt.px_per_m
    out[:, 1] = belt.center_row + pts[:, 1] * belt.px_per_m
    return out


def item_box_px(belt: BeltConfig, item: Item) -> tuple[float, float, float, float]:
    poly = item_polygon_px(belt, item)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


def item_roi(belt: BeltConfig, item: Item, margin_px: int = 8) -> tuple[int, int, int, int] | None:
    """Clipped ``(r0, r1, c0, c1)`` window around the item, or ``None`` if off-frame.

    Everything in this module works inside this window rather than on the full
    frame; that is what keeps dataset generation at a few milliseconds a frame.
    """
    x0, y0, x1, y1 = item_box_px(belt, item)
    n = belt.image_px
    c0 = int(max(0, np.floor(x0) - margin_px))
    c1 = int(min(n, np.ceil(x1) + margin_px))
    r0 = int(max(0, np.floor(y0) - margin_px))
    r1 = int(min(n, np.ceil(y1) + margin_px))
    if c1 - c0 < 1 or r1 - r0 < 1:
        return None
    return r0, r1, c0, c1


def _local_mask(belt: BeltConfig, item: Item, roi: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = roi
    mask = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    poly = item_polygon_px(belt, item)
    poly = poly - np.array([c0, r0], dtype=np.float64)
    if item.material.round_shape:
        center = (float(poly[:, 0].mean()), float(poly[:, 1].mean()))
        axes = (
            max(1.0, 0.5 * item.long_m * belt.px_per_m),
            max(1.0, 0.5 * item.short_m * belt.px_per_m),
        )
        cv2.ellipse(
            mask,
            (int(round(center[0])), int(round(center[1]))),
            (int(round(axes[0])), int(round(axes[1]))),
            float(np.degrees(item.theta)),
            0,
            360,
            255,
            -1,
        )
    else:
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
    return mask


# --------------------------------------------------------------------------
# RGB
# --------------------------------------------------------------------------
def _belt_background(rng: np.random.Generator, belt: BeltConfig) -> np.ndarray:
    n = belt.image_px
    img = np.full((n, n, 3), (46, 48, 52), dtype=np.uint8)
    # side rails outside the belt
    half = belt.belt_half_px
    top = int(round(belt.center_row - half))
    bot = int(round(belt.center_row + half))
    img[:top] = (74, 78, 84)
    img[bot:] = (74, 78, 84)
    cv2.line(img, (0, top), (n, top), (96, 100, 108), 2)
    cv2.line(img, (0, bot), (n, bot), (96, 100, 108), 2)
    # belt ribbing
    for col in range(0, n, 14):
        shade = int(rng.integers(52, 62))
        cv2.line(img, (col, top), (col, bot), (shade, shade + 2, shade + 5), 1)
    grain = rng.normal(0.0, 5.0, size=(n, n, 1)).astype(np.float32)
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def _apply_lighting(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7) * n, rng.uniform(0.3, 0.7) * n
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n * 0.9)
    gain = (1.12 - 0.34 * r**2)[..., None]
    out = img.astype(np.float32) * gain
    out += rng.normal(0.0, 3.2, size=img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_rgb(rng: np.random.Generator, scene: Scene) -> np.ndarray:
    belt = scene.belt
    img = _belt_background(rng, belt)
    for item in scene.visible():
        roi = item_roi(belt, item)
        if roi is None:
            continue
        r0, r1, c0, c1 = roi
        mask = _local_mask(belt, item, roi)
        if mask.max() == 0:
            continue
        view = img[r0:r1, c0:c1]
        sel = mask > 0
        texture = rng.normal(0.0, 7.0, size=(*mask.shape, 1)).astype(np.float32)
        layer = np.clip(np.array(item.color, np.float32) + texture, 0, 255).astype(np.uint8)
        view[sel] = layer[sel]
        # contact shadow: darken the rim just outside the silhouette
        rim = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - mask
        rim_sel = rim > 0
        if rim_sel.any():
            view[rim_sel] = (view[rim_sel].astype(np.float32) * 0.72).astype(np.uint8)
        if item.material.specular > 0.0:
            spec = np.zeros(mask.shape, dtype=np.float32)
            poly = item_polygon_px(belt, item) - np.array([c0, r0], dtype=np.float64)
            cx = float(poly[:, 0].mean() + rng.normal(0, 2.5))
            cy = float(poly[:, 1].mean() + rng.normal(0, 2.5))
            axis = max(1.5, 0.22 * item.long_m * belt.px_per_m)
            cv2.circle(spec, (int(cx), int(cy)), int(axis), 1.0, -1)
            spec = cv2.GaussianBlur(spec, (0, 0), axis * 0.7)
            spec *= item.material.specular * 210.0
            spec[~sel] = 0.0
            img[r0:r1, c0:c1] = np.clip(view.astype(np.float32) + spec[..., None], 0, 255).astype(
                np.uint8
            )
    return _apply_lighting(rng, img)


# --------------------------------------------------------------------------
# LWIR
# --------------------------------------------------------------------------
def render_thermal(rng: np.random.Generator, scene: Scene, cfg: ThermalConfig) -> np.ndarray:
    """Apparent temperature rise above ambient, in kelvin, float32."""
    belt = scene.belt
    n = belt.image_px
    shape = (n, n)
    refl_dt = cfg.reflected_ambient_c - belt.ambient_c

    field = np.full(shape, belt.surface_dt, dtype=np.float32)
    # slow spatial drift of the belt surface temperature
    coarse = rng.normal(0.0, 0.45, size=(8, 8)).astype(np.float32)
    field += cv2.resize(coarse, shape, interpolation=cv2.INTER_CUBIC)

    for item in scene.visible():
        roi = item_roi(belt, item, margin_px=18 if item.dt_core > 8.0 else 6)
        if roi is None:
            continue
        r0, r1, c0, c1 = roi
        mask = _local_mask(belt, item, roi)
        if mask.max() == 0:
            continue
        # hottest at the core, falling off toward the rim
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        peak = float(dist.max())
        profile = dist / (peak + _EPS)
        profile = 0.45 + 0.55 * np.sqrt(np.clip(profile, 0.0, 1.0))
        apparent = item.emissivity * item.dt_core * profile + (1.0 - item.emissivity) * refl_dt
        sel = mask > 0
        # thermal bleed into the belt around a genuinely hot item
        if item.dt_core > 8.0:
            halo = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 6.0)
            field[r0:r1, c0:c1] += halo * float(item.emissivity * item.dt_core) * 0.16
        field[r0:r1, c0:c1][sel] = apparent[sel].astype(np.float32)

    field = cv2.GaussianBlur(field, (0, 0), cfg.blur_sigma_px)
    field += rng.normal(0.0, cfg.noise_equivalent_dt, size=shape).astype(np.float32)
    field += rng.normal(0.0, cfg.fixed_pattern_sigma, size=(1, n)).astype(np.float32)
    return field.astype(np.float32)


# --------------------------------------------------------------------------
# frame bundle
# --------------------------------------------------------------------------
def render_frame(
    rng: np.random.Generator, scene: Scene, thermal_cfg: ThermalConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rgb_uint8_hwc, thermal_dt_float32)``."""
    return render_rgb(rng, scene), render_thermal(rng, scene, thermal_cfg)


def ground_truth(
    scene: Scene, *, min_side_px: float = 3.0
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Boxes ``(N, 4)`` in render pixels, class ids ``(N,)`` and item uids."""
    from ..config import CLASS_INDEX

    belt = scene.belt
    boxes: list[list[float]] = []
    labels: list[int] = []
    uids: list[int] = []
    limit = belt.image_px - 1.0
    for item in scene.visible():
        if item.label is None:
            continue
        x0, y0, x1, y1 = item_box_px(belt, item)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(limit, x1), min(limit, y1)
        if (x1 - x0) < min_side_px or (y1 - y0) < min_side_px:
            continue
        boxes.append([x0, y0, x1, y1])
        labels.append(CLASS_INDEX[item.label])
        uids.append(item.uid)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), []
    return np.array(boxes, np.float32), np.array(labels, np.int64), uids


def to_network_input(
    rgb: np.ndarray, thermal: np.ndarray, det: DetectorConfig, thermal_cfg: ThermalConfig
) -> np.ndarray:
    """Build the 4-channel ``(4, H, W)`` float32 tensor the detector expects."""
    size = (det.input_px, det.input_px)
    rgb_small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    th_small = cv2.resize(thermal, size, interpolation=cv2.INTER_AREA)
    th_norm = np.clip(th_small / thermal_cfg.span_k, -0.2, 1.2).astype(np.float32)
    stacked = np.concatenate([rgb_small.transpose(2, 0, 1), th_norm[None]], axis=0)
    return np.ascontiguousarray(stacked, dtype=np.float32)


def thermal_to_color(
    thermal: np.ndarray, span_k: float = 30.0, *, baseline: float | None = None, gamma: float = 0.75
) -> np.ndarray:
    """Inferno false-colour view of the LWIR frame, for the console and video.

    The display span is deliberately narrower than the network's input span:
    a self-heating cell sits 5-15 K above the belt, and stretching the palette
    over the full 60 K runaway range would render exactly the cue an operator
    needs as near-black. Scaling is relative to the belt baseline so the view
    does not wash out when the hall warms up.
    """
    base = float(np.median(thermal)) if baseline is None else baseline
    norm = np.clip((thermal - base) / span_k, 0.0, 1.0) ** gamma
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
