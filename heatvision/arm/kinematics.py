"""Forward and inverse kinematics for the 2R SCARA arm (+ wrist, + Z).

Frame convention: everything is in *belt* coordinates, metres. ``x`` runs
downstream along the belt, ``y`` runs across it, and the arm base sits beside
the belt at ``ArmConfig.base_xy_m``. The wrist axis is vertical, so the
gripper orientation is a single angle in the belt plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ArmConfig


class UnreachableError(ValueError):
    """Raised when a Cartesian target lies outside the arm's workspace."""


@dataclass(frozen=True)
class JointState:
    """Shoulder, elbow, wrist (rad) and the vertical axis (m)."""

    q1: float
    q2: float
    q3: float
    z: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.q1, self.q2, self.q3, self.z)

    @property
    def arm_joints(self) -> tuple[float, float, float]:
        return (self.q1, self.q2, self.q3)


def forward(cfg: ArmConfig, q1: float, q2: float) -> tuple[float, float]:
    """Tool centre point in belt metres for the two planar joints."""
    bx, by = cfg.base_xy_m
    x = bx + cfg.link1_m * math.cos(q1) + cfg.link2_m * math.cos(q1 + q2)
    y = by + cfg.link1_m * math.sin(q1) + cfg.link2_m * math.sin(q1 + q2)
    return x, y


def elbow_position(cfg: ArmConfig, q1: float) -> tuple[float, float]:
    bx, by = cfg.base_xy_m
    return bx + cfg.link1_m * math.cos(q1), by + cfg.link1_m * math.sin(q1)


def reach_limits(cfg: ArmConfig) -> tuple[float, float]:
    """``(min_radius, max_radius)`` of the annular workspace."""
    return abs(cfg.link1_m - cfg.link2_m) + 0.02, cfg.link1_m + cfg.link2_m - 0.01


def is_reachable(cfg: ArmConfig, x: float, y: float) -> bool:
    bx, by = cfg.base_xy_m
    r = math.hypot(x - bx, y - by)
    r_min, r_max = reach_limits(cfg)
    return r_min <= r <= r_max


def inverse(
    cfg: ArmConfig,
    x: float,
    y: float,
    *,
    tool_angle: float = 0.0,
    prefer: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Solve for ``(q1, q2, q3)``.

    Both elbow configurations are computed; the one closest to ``prefer`` (the
    current pose) is returned, which avoids the arm flipping through a
    singularity between two nearby picks.

    ``tool_angle`` is the desired gripper orientation in the belt plane; the
    wrist joint absorbs whatever the first two joints leave over.
    """
    bx, by = cfg.base_xy_m
    dx, dy = x - bx, y - by
    r = math.hypot(dx, dy)
    r_min, r_max = reach_limits(cfg)
    if not (r_min <= r <= r_max):
        raise UnreachableError(
            f"target ({x:.3f}, {y:.3f}) at r={r:.3f} m outside workspace [{r_min:.3f}, {r_max:.3f}]"
        )

    l1, l2 = cfg.link1_m, cfg.link2_m
    cos_q2 = (r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2_mag = math.acos(cos_q2)

    solutions: list[tuple[float, float, float]] = []
    for sign in (1.0, -1.0):
        q2 = sign * q2_mag
        k1 = l1 + l2 * math.cos(q2)
        k2 = l2 * math.sin(q2)
        q1 = math.atan2(dy, dx) - math.atan2(k2, k1)
        q3 = wrap_angle(tool_angle - (q1 + q2))
        solutions.append((wrap_angle(q1), wrap_angle(q2), q3))

    if prefer is None:
        return solutions[0]
    return min(solutions, key=lambda s: joint_distance(s, prefer))


def wrap_angle(a: float) -> float:
    """Wrap to ``(-pi, pi]``."""
    return math.atan2(math.sin(a), math.cos(a))


def angle_delta(a: float, b: float) -> float:
    """Shortest signed rotation from ``b`` to ``a``."""
    return wrap_angle(a - b)


def joint_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(abs(angle_delta(x, y)) for x, y in zip(a, b))


def grasp_angle(item_theta: float) -> float:
    """Gripper angle that closes across an item's *short* axis.

    Jaws must straddle the narrow dimension of a cell, never squeeze it along
    its length, and never come down on the terminals.
    """
    return wrap_angle(item_theta + math.pi / 2)
