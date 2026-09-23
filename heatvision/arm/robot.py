"""Kinematic simulation of the removal arm, with a real pick state machine.

Nothing here is scripted. The arm solves an intercept for a target that is
still moving, drives a trajectory to it under acceleration limits, closes the
gripper, and then *checks whether it actually got the thing*: at the moment
the jaws close, the tool-to-item error is measured and the grasp fails if the
error exceeds the jaw opening. Missed picks are therefore real failures that
show up in the end-to-end numbers, which is the only way those numbers mean
anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from ..config import ArmConfig
from .kinematics import JointState, UnreachableError, forward, inverse, is_reachable
from .trajectory import JointTrajectory, trapezoid_duration


class ArmState(str, Enum):
    IDLE = "idle"
    TO_INTERCEPT = "to_intercept"
    DESCEND = "descend"
    GRIP = "grip"
    LIFT = "lift"
    TO_BIN = "to_bin"
    RELEASE = "release"
    RETURN = "return"


@dataclass
class PickJob:
    """One extraction request handed to the arm."""

    track_id: int
    target_xy: tuple[float, float]
    tool_angle: float
    bin_xy: tuple[float, float]
    grip_force_n: float
    #: belt speed to assume while planning the intercept (0.0 = line stopped)
    belt_speed_mps: float
    label: str = ""
    gt_uid: int | None = None


@dataclass
class PickResult:
    track_id: int
    success: bool
    reason: str
    duration_s: float
    grasp_error_m: float | None = None
    gt_uid: int | None = None


@dataclass
class Arm:
    """4-DOF arm (shoulder, elbow, wrist, Z) with a two-finger gripper."""

    cfg: ArmConfig
    #: returns the item's current ``(x, y)`` or ``None`` if it is gone
    locate: Callable[[PickJob], tuple[float, float] | None] | None = None
    #: called on a successful grasp so the world can mark the item removed
    on_grasp: Callable[[PickJob], None] | None = None
    #: jaw opening: the grasp fails beyond this tool-to-item error
    grasp_tolerance_m: float = 0.026

    state: ArmState = ArmState.IDLE
    joints: JointState = field(init=False)
    job: PickJob | None = None
    holding: bool = False
    grip_force_n: float = 0.0
    _traj: JointTrajectory | None = field(default=None, repr=False)
    _t: float = 0.0
    _job_t: float = 0.0
    _home: tuple[float, float, float, float] = field(init=False, repr=False)
    _last_error: float | None = None

    def __post_init__(self) -> None:
        home_xy = (self.cfg.base_xy_m[0], self.cfg.base_xy_m[1] + 0.35)
        q1, q2, q3 = inverse(self.cfg, *home_xy, tool_angle=0.0)
        self._home = (q1, q2, q3, self.cfg.z_clear_m)
        self.joints = JointState(q1, q2, q3, self.cfg.z_clear_m)

    # ------------------------------------------------------------------ info
    @property
    def busy(self) -> bool:
        return self.state is not ArmState.IDLE

    @property
    def tool_xy(self) -> tuple[float, float]:
        return forward(self.cfg, self.joints.q1, self.joints.q2)

    @property
    def limits(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        v = (*self.cfg.joint_vel_limit, self.cfg.z_vel_limit)
        a = (*self.cfg.joint_acc_limit, self.cfg.z_acc_limit)
        return v, a

    def can_reach(self, x: float, y: float) -> bool:
        return is_reachable(self.cfg, x, y)

    # ------------------------------------------------------------- planning
    def _goto(self, q: tuple[float, float, float], z: float, *, min_duration: float = 0.0) -> None:
        v, a = self.limits
        self._traj = JointTrajectory.plan(self.joints.as_tuple(), (*q, z), v, a, min_duration=min_duration)
        self._t = 0.0

    def time_to(self, x: float, y: float, z: float, tool_angle: float = 0.0) -> float | None:
        """Duration of a move to a Cartesian pose, or ``None`` if unreachable.

        ``tool_angle`` matters: on a wide re-orientation the wrist, not the
        shoulder, sets the segment duration, and leaving it out of the estimate
        makes the intercept land behind a moving target.
        """
        try:
            q = inverse(self.cfg, x, y, tool_angle=tool_angle, prefer=self.joints.arm_joints)
        except UnreachableError:
            return None
        v, a = self.limits
        return JointTrajectory.plan(self.joints.as_tuple(), (*q, z), v, a).duration

    @property
    def _descend_and_grip_s(self) -> float:
        """Time from arriving above the target to the jaws being shut."""
        return (
            trapezoid_duration(
                self.cfg.z_clear_m - self.cfg.z_pick_m, self.cfg.z_vel_limit, self.cfg.z_acc_limit
            )
            + self.cfg.grip_time_s
        )

    def solve_intercept(self, job: PickJob, *, iterations: int = 6) -> tuple[float, float] | None:
        """Where to meet a target that keeps moving while the arm flies to it.

        Fixed-point iteration: guess the total time to jaw-close, advance the
        target by that much, re-solve the flight time to the new point, repeat.
        Converges in a handful of steps because belt speed is small next to
        joint speed.

        The budget deliberately includes the descent and the gripper dwell, not
        just the flight: aiming at where the item is when the arm *arrives*
        would close the jaws a descent-time behind it, which is precisely the
        classic pick-on-the-fly failure.
        """
        x, y = job.target_xy
        extra = self._descend_and_grip_s
        t_total = extra
        for _ in range(iterations):
            px = x + job.belt_speed_mps * t_total
            if not self.can_reach(px, y):
                return None
            flight = self.time_to(px, y, self.cfg.z_clear_m, job.tool_angle)
            if flight is None:
                return None
            new_total = flight + extra
            converged = abs(new_total - t_total) < 1e-3
            t_total = new_total
            if converged:
                break
        px = x + job.belt_speed_mps * t_total
        if not self.can_reach(px, y) or px > self.cfg.base_xy_m[0] + 0.55:
            return None
        return px, y

    def start(self, job: PickJob) -> bool:
        """Begin a pick. Returns ``False`` if the target cannot be intercepted."""
        if self.busy:
            return False
        intercept = self.solve_intercept(job)
        if intercept is None:
            return False
        try:
            q = inverse(self.cfg, *intercept, tool_angle=job.tool_angle, prefer=self.joints.arm_joints)
        except UnreachableError:
            return False
        self.job = job
        self.grip_force_n = job.grip_force_n
        self.state = ArmState.TO_INTERCEPT
        self._job_t = 0.0
        self._last_error = None
        self._goto(q, self.cfg.z_clear_m)
        return True

    def abort(self, reason: str = "aborted") -> PickResult | None:
        if self.job is None:
            return None
        result = PickResult(self.job.track_id, False, reason, self._job_t, gt_uid=self.job.gt_uid)
        self._finish()
        return result

    def _finish(self) -> None:
        self.job = None
        self.holding = False
        self.grip_force_n = 0.0
        self.state = ArmState.RETURN
        self._goto(self._home[:3], self._home[3])

    # ------------------------------------------------------------- stepping
    def step(self, dt: float) -> PickResult | None:
        """Advance the arm by ``dt``. Returns a result when a pick concludes."""
        if self.state is ArmState.IDLE:
            return None
        self._t += dt
        self._job_t += dt
        if self._traj is not None:
            q1, q2, q3, z = self._traj.at(self._t)
            self.joints = JointState(q1, q2, q3, z)
            if not self._traj.is_done(self._t):
                return None
        return self._on_segment_done()

    def _on_segment_done(self) -> PickResult | None:
        job = self.job
        if self.state is ArmState.RETURN:
            self.state = ArmState.IDLE
            self._traj = None
            return None
        if job is None:
            self.state = ArmState.IDLE
            return None

        if self.state is ArmState.TO_INTERCEPT:
            self.state = ArmState.DESCEND
            self._goto(self.joints.arm_joints, self.cfg.z_pick_m)
            return None

        if self.state is ArmState.DESCEND:
            self.state = ArmState.GRIP
            self._traj = None
            self._t = 0.0
            return None

        if self.state is ArmState.GRIP:
            # dwell for the jaw-close time, then verify we actually have it
            if self._t < self.cfg.grip_time_s:
                return None
            actual = self.locate(job) if self.locate else job.target_xy
            if actual is None:
                result = PickResult(job.track_id, False, "target_gone", self._job_t, gt_uid=job.gt_uid)
                self._finish()
                return result
            tx, ty = self.tool_xy
            error = math.hypot(actual[0] - tx, actual[1] - ty)
            self._last_error = error
            if error > self.grasp_tolerance_m:
                result = PickResult(
                    job.track_id, False, "grasp_miss", self._job_t, grasp_error_m=error, gt_uid=job.gt_uid
                )
                self._finish()
                return result
            self.holding = True
            if self.on_grasp:
                self.on_grasp(job)
            self.state = ArmState.LIFT
            self._goto(self.joints.arm_joints, self.cfg.z_clear_m)
            return None

        if self.state is ArmState.LIFT:
            try:
                q = inverse(self.cfg, *job.bin_xy, prefer=self.joints.arm_joints)
            except UnreachableError:
                result = PickResult(
                    job.track_id, False, "bin_unreachable", self._job_t, gt_uid=job.gt_uid
                )
                self._finish()
                return result
            self.state = ArmState.TO_BIN
            self._goto(q, self.cfg.z_clear_m)
            return None

        if self.state is ArmState.TO_BIN:
            self.state = ArmState.RELEASE
            self._traj = None
            self._t = 0.0
            return None

        if self.state is ArmState.RELEASE:
            if self._t < self.cfg.release_time_s:
                return None
            result = PickResult(
                job.track_id,
                True,
                "placed",
                self._job_t,
                grasp_error_m=self._last_error,
                gt_uid=job.gt_uid,
            )
            self._finish()
            return result

        self.state = ArmState.IDLE
        return None
