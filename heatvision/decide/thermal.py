"""Calibrated thermal measurements taken inside a detection box.

The detector says *what* an object is. This module says *how hot it is right
now*, in kelvin above the belt baseline, which is the quantity the safety
policy actually reasons about. Keeping the two separate matters: a network
score of 0.9 must never be allowed to overrule a 50 K thermal reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalReading:
    """Thermal state of one region of interest, relative to the belt."""

    peak_dt: float  # K above baseline, 98th percentile inside the box
    mean_dt: float  # K above baseline, median inside the box
    hot_area_frac: float  # fraction of the box more than 8 K above baseline
    baseline_dt: float  # the belt baseline this reading was taken against

    @property
    def is_warm(self) -> bool:
        return self.peak_dt >= 6.0


def belt_baseline(thermal: np.ndarray) -> float:
    """Robust belt-surface temperature.

    The belt is by far the most common surface in the frame, so the median is
    a solid estimate that is immune to a few very hot objects. This is what
    makes the policy thresholds independent of ambient temperature — a plant
    that runs at 35 C in summer needs no re-tuning.
    """
    return float(np.median(thermal))


def _shrink(box: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw, hh = 0.5 * (x1 - x0) * factor, 0.5 * (y1 - y0) * factor
    return cx - hw, cy - hh, cx + hw, cy + hh


def measure(
    thermal: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    baseline: float | None = None,
    shrink: float = 0.82,
    hot_k: float = 8.0,
) -> ThermalReading:
    """Sample ``thermal`` inside ``box`` (render pixels).

    The box is shrunk slightly first so the rim — which mixes object and belt —
    does not drag the reading toward the baseline.
    """
    base = belt_baseline(thermal) if baseline is None else baseline
    h, w = thermal.shape
    x0, y0, x1, y1 = _shrink(box, shrink)
    c0 = int(np.clip(np.floor(x0), 0, w - 1))
    c1 = int(np.clip(np.ceil(x1), c0 + 1, w))
    r0 = int(np.clip(np.floor(y0), 0, h - 1))
    r1 = int(np.clip(np.ceil(y1), r0 + 1, h))
    patch = thermal[r0:r1, c0:c1]
    if patch.size == 0:
        return ThermalReading(0.0, 0.0, 0.0, base)
    rel = patch - base
    return ThermalReading(
        peak_dt=float(np.percentile(rel, 98)),
        mean_dt=float(np.median(rel)),
        hot_area_frac=float(np.mean(rel > hot_k)),
        baseline_dt=base,
    )


def rise_rate(history: list[tuple[float, float]], *, min_points: int = 5, window_s: float = 2.0) -> float:
    """Least-squares K/s slope over the recent ``(time_s, peak_dt)`` history.

    A runaway cell is defined by its *trend*, not its absolute temperature: a
    35 K brake disc is cooling, a 35 K pouch that gained 12 K in three seconds
    is venting. Returns 0.0 until there are enough points to be meaningful.
    """
    if len(history) < min_points:
        return 0.0
    t_end = history[-1][0]
    pts = [(t, v) for t, v in history if t >= t_end - window_s]
    if len(pts) < min_points:
        pts = history[-min_points:]
    t = np.array([p[0] for p in pts], dtype=np.float64)
    v = np.array([p[1] for p in pts], dtype=np.float64)
    t = t - t.mean()
    denom = float((t * t).sum())
    if denom < 1e-9:
        return 0.0
    return float((t * (v - v.mean())).sum() / denom)
