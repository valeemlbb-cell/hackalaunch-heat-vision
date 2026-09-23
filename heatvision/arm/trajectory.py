"""Time-optimal trapezoidal joint trajectories with synchronised arrival.

Each joint gets a trapezoidal (or triangular, for short moves) velocity
profile inside its own limits; the slowest joint sets the segment duration and
the others are stretched to match, so the tool follows a repeatable path
instead of a different curve every time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def trapezoid_duration(distance: float, v_max: float, a_max: float) -> float:
    """Shortest time to move ``distance`` from rest to rest."""
    d = abs(distance)
    if d < 1e-9:
        return 0.0
    d_ramp = v_max * v_max / a_max  # distance used by a full accel + decel
    if d <= d_ramp:  # triangular profile, never reaches v_max
        return 2.0 * math.sqrt(d / a_max)
    return v_max / a_max + d / v_max


def trapezoid_position(t: float, distance: float, v_max: float, a_max: float, duration: float) -> float:
    """Position along a profile stretched to exactly ``duration`` seconds."""
    d = abs(distance)
    if d < 1e-9 or duration <= 0.0:
        return distance
    sign = 1.0 if distance >= 0 else -1.0
    t = max(0.0, min(t, duration))
    # stretch: solve for the peak velocity that covers d in `duration`
    disc = a_max * a_max * duration * duration - 4.0 * a_max * d
    if disc < 0.0:  # duration too short for the limits: fall back to triangular
        t_acc = duration / 2.0
        v_peak = d / t_acc
        a = v_peak / t_acc
        if t <= t_acc:
            return sign * 0.5 * a * t * t
        tt = t - t_acc
        return sign * (0.5 * a * t_acc * t_acc + v_peak * tt - 0.5 * a * tt * tt)
    v_peak = 0.5 * (a_max * duration - math.sqrt(disc))
    t_acc = v_peak / a_max
    t_flat = duration - 2.0 * t_acc
    if t <= t_acc:
        s = 0.5 * a_max * t * t
    elif t <= t_acc + t_flat:
        s = 0.5 * v_peak * t_acc + v_peak * (t - t_acc)
    else:
        td = t - t_acc - t_flat
        s = 0.5 * v_peak * t_acc + v_peak * t_flat + v_peak * td - 0.5 * a_max * td * td
    return sign * min(s, d)


@dataclass
class JointTrajectory:
    """Synchronised multi-axis move from ``start`` to ``goal``."""

    start: tuple[float, ...]
    goal: tuple[float, ...]
    v_max: tuple[float, ...]
    a_max: tuple[float, ...]
    duration: float

    @classmethod
    def plan(
        cls,
        start: tuple[float, ...],
        goal: tuple[float, ...],
        v_max: tuple[float, ...],
        a_max: tuple[float, ...],
        *,
        min_duration: float = 0.0,
    ) -> "JointTrajectory":
        if not (len(start) == len(goal) == len(v_max) == len(a_max)):
            raise ValueError("start, goal and limits must have the same length")
        durations = [
            trapezoid_duration(g - s, v, a) for s, g, v, a in zip(start, goal, v_max, a_max)
        ]
        duration = max([min_duration, *durations]) if durations else min_duration
        return cls(tuple(start), tuple(goal), tuple(v_max), tuple(a_max), duration)

    def at(self, t: float) -> tuple[float, ...]:
        if self.duration <= 0.0:
            return self.goal
        out = []
        for s, g, v, a in zip(self.start, self.goal, self.v_max, self.a_max):
            out.append(s + trapezoid_position(t, g - s, v, a, self.duration))
        return tuple(out)

    def is_done(self, t: float) -> bool:
        return t >= self.duration - 1e-9
