"""Kinematics, trajectories and the pick state machine."""

from __future__ import annotations

import math

import pytest

from heatvision.arm.kinematics import (
    ArmConfig,
    UnreachableError,
    forward,
    grasp_angle,
    inverse,
    is_reachable,
    reach_limits,
    wrap_angle,
)
from heatvision.arm.robot import Arm, ArmState, PickJob
from heatvision.arm.trajectory import JointTrajectory, trapezoid_duration, trapezoid_position

CFG = ArmConfig()


class TestKinematics:
    @pytest.mark.parametrize(
        "target",
        [(0.55, 0.10), (0.30, -0.10), (0.80, 0.20), (0.55, 0.35), (0.20, -0.30)],
    )
    def test_inverse_then_forward_returns_the_target(self, target):
        q1, q2, _q3 = inverse(CFG, *target)
        assert forward(CFG, q1, q2) == pytest.approx(target, abs=1e-9)

    def test_both_elbow_solutions_reach_the_same_point(self):
        target = (0.6, 0.1)
        near = inverse(CFG, *target, prefer=(0.0, 1.0, 0.0))
        far = inverse(CFG, *target, prefer=(0.0, -1.0, 0.0))
        assert forward(CFG, *near[:2]) == pytest.approx(forward(CFG, *far[:2]), abs=1e-9)
        assert near[1] * far[1] < 0  # genuinely different elbow signs

    def test_prefer_picks_the_closer_configuration(self):
        current = inverse(CFG, 0.62, 0.05)
        chosen = inverse(CFG, 0.60, 0.05, prefer=current)
        assert abs(wrap_angle(chosen[1] - current[1])) < 0.6

    def test_out_of_reach_raises(self):
        with pytest.raises(UnreachableError):
            inverse(CFG, 2.5, 2.5)

    def test_reach_limits_are_consistent_with_is_reachable(self):
        r_min, r_max = reach_limits(CFG)
        bx, by = CFG.base_xy_m
        assert is_reachable(CFG, bx + r_max - 0.01, by)
        assert not is_reachable(CFG, bx + r_max + 0.05, by)
        assert not is_reachable(CFG, bx + r_min - 0.01, by)

    def test_the_whole_belt_width_is_reachable_at_the_pick_line(self):
        for y in (-0.44, -0.2, 0.0, 0.2, 0.44):
            assert is_reachable(CFG, CFG.base_xy_m[0], y)

    def test_tool_angle_is_honoured_by_the_wrist(self):
        q1, q2, q3 = inverse(CFG, 0.6, 0.1, tool_angle=1.0)
        assert wrap_angle(q1 + q2 + q3) == pytest.approx(1.0, abs=1e-9)

    def test_grasp_angle_is_perpendicular_to_the_long_axis(self):
        assert abs(wrap_angle(grasp_angle(0.4) - 0.4)) == pytest.approx(math.pi / 2, abs=1e-9)


class TestTrajectory:
    def test_zero_distance_takes_no_time(self):
        assert trapezoid_duration(0.0, 1.0, 1.0) == 0.0

    def test_triangular_profile_for_a_short_move(self):
        # d < v^2/a so the profile never reaches v_max
        assert trapezoid_duration(0.1, 2.0, 4.0) == pytest.approx(2 * math.sqrt(0.1 / 4.0))

    def test_trapezoidal_profile_for_a_long_move(self):
        assert trapezoid_duration(10.0, 2.0, 4.0) == pytest.approx(2.0 / 4.0 + 10.0 / 2.0)

    def test_longer_distance_never_takes_less_time(self):
        prev = 0.0
        for d in (0.1, 0.5, 1.0, 5.0, 20.0):
            now = trapezoid_duration(d, 2.0, 3.0)
            assert now >= prev
            prev = now

    def test_position_endpoints_are_exact(self):
        d = trapezoid_duration(1.5, 2.0, 4.0)
        assert trapezoid_position(0.0, 1.5, 2.0, 4.0, d) == pytest.approx(0.0, abs=1e-9)
        assert trapezoid_position(d, 1.5, 2.0, 4.0, d) == pytest.approx(1.5, abs=1e-6)

    def test_position_is_monotone_and_bounded(self):
        d = trapezoid_duration(1.5, 2.0, 4.0)
        prev = -1.0
        for i in range(51):
            s = trapezoid_position(d * i / 50, 1.5, 2.0, 4.0, d)
            assert s >= prev - 1e-9
            assert -1e-9 <= s <= 1.5 + 1e-6
            prev = s

    def test_negative_moves_mirror_positive_ones(self):
        d = trapezoid_duration(1.5, 2.0, 4.0)
        assert trapezoid_position(d * 0.4, -1.5, 2.0, 4.0, d) == pytest.approx(
            -trapezoid_position(d * 0.4, 1.5, 2.0, 4.0, d)
        )

    def test_axes_are_synchronised_to_the_slowest_one(self):
        traj = JointTrajectory.plan((0, 0), (2.0, 0.05), (1.0, 1.0), (2.0, 2.0))
        assert traj.duration == pytest.approx(trapezoid_duration(2.0, 1.0, 2.0))
        assert traj.at(traj.duration) == pytest.approx((2.0, 0.05), abs=1e-6)

    def test_stretched_axis_respects_its_velocity_limit(self):
        traj = JointTrajectory.plan((0, 0), (2.0, 0.05), (1.0, 1.0), (2.0, 2.0))
        dt = traj.duration / 400
        for i in range(400):
            a = traj.at(i * dt)[1]
            b = traj.at((i + 1) * dt)[1]
            assert abs(b - a) / dt <= 1.0 + 1e-6

    def test_min_duration_can_slow_a_move_down(self):
        traj = JointTrajectory.plan((0.0,), (1.0,), (5.0,), (10.0,), min_duration=3.0)
        assert traj.duration == pytest.approx(3.0)
        assert traj.at(3.0)[0] == pytest.approx(1.0, abs=1e-6)


class TestArmPickCycle:
    def _run(self, arm: Arm, job: PickJob, *, max_s: float = 12.0, on_step=None):
        assert arm.start(job) is True
        t = 0.0
        while t < max_s:
            if on_step is not None:
                on_step(t)
            result = arm.step(CFG.dt)
            t += CFG.dt
            if result is not None:
                return result, t
        raise AssertionError("pick did not finish")

    def test_a_static_target_is_picked_and_placed(self):
        target = (0.55, 0.10)
        grabbed: list[PickJob] = []
        arm = Arm(CFG, locate=lambda _job: target, on_grasp=grabbed.append)
        job = PickJob(1, target, 0.0, CFG.quarantine_bin_m, 12.0, 0.0, "cell", gt_uid=5)
        result, _t = self._run(arm, job)
        assert result.success is True
        assert result.reason == "placed"
        assert len(grabbed) == 1
        assert result.grasp_error_m < arm.grasp_tolerance_m

    def test_the_arm_visits_every_state_in_order(self):
        target = (0.55, 0.10)
        arm = Arm(CFG, locate=lambda _job: target)
        job = PickJob(1, target, 0.0, CFG.quarantine_bin_m, 12.0, 0.0)
        seen: list[ArmState] = []
        assert arm.start(job)
        for _ in range(900):
            if not seen or seen[-1] is not arm.state:
                seen.append(arm.state)
            if arm.step(CFG.dt) is not None:
                break
        assert seen[:6] == [
            ArmState.TO_INTERCEPT,
            ArmState.DESCEND,
            ArmState.GRIP,
            ArmState.LIFT,
            ArmState.TO_BIN,
            ArmState.RELEASE,
        ]

    def test_a_moving_target_is_intercepted(self):
        state = {"x": 0.40}
        arm = Arm(CFG, locate=lambda _job: (state["x"], 0.08))
        job = PickJob(1, (state["x"], 0.08), 0.0, CFG.quarantine_bin_m, 12.0, 0.35, gt_uid=1)

        def advance(_t: float) -> None:
            if arm.state in (ArmState.TO_INTERCEPT, ArmState.DESCEND, ArmState.GRIP):
                state["x"] += 0.35 * CFG.dt

        result, _t = self._run(arm, job, on_step=advance)
        assert result.success is True, result.reason

    def test_a_target_that_is_yanked_away_produces_a_real_miss(self):
        arm = Arm(CFG, locate=lambda _job: (0.20, -0.30))
        job = PickJob(1, (0.55, 0.10), 0.0, CFG.quarantine_bin_m, 12.0, 0.0, gt_uid=2)
        result, _t = self._run(arm, job)
        assert result.success is False
        assert result.reason == "grasp_miss"
        assert result.grasp_error_m > arm.grasp_tolerance_m

    def test_a_vanished_target_is_reported_not_grasped(self):
        arm = Arm(CFG, locate=lambda _job: None)
        job = PickJob(1, (0.55, 0.10), 0.0, CFG.quarantine_bin_m, 12.0, 0.0, gt_uid=3)
        result, _t = self._run(arm, job)
        assert result.success is False
        assert result.reason == "target_gone"

    def test_an_unreachable_target_is_refused_up_front(self):
        arm = Arm(CFG)
        job = PickJob(1, (3.0, 3.0), 0.0, CFG.quarantine_bin_m, 12.0, 0.0)
        assert arm.start(job) is False
        assert arm.state is ArmState.IDLE

    def test_the_arm_refuses_a_second_job_while_busy(self):
        arm = Arm(CFG, locate=lambda _job: (0.55, 0.10))
        job = PickJob(1, (0.55, 0.10), 0.0, CFG.quarantine_bin_m, 12.0, 0.0)
        assert arm.start(job) is True
        assert arm.start(job) is False

    def test_transit_stays_at_the_safe_height_until_the_descent(self):
        target = (0.55, 0.10)
        arm = Arm(CFG, locate=lambda _job: target)
        job = PickJob(1, target, 0.0, CFG.quarantine_bin_m, 12.0, 0.0)
        arm.start(job)
        while arm.state is ArmState.TO_INTERCEPT:
            arm.step(CFG.dt)
            assert arm.joints.z >= CFG.z_clear_m - 1e-9

    def test_abort_releases_the_arm(self):
        arm = Arm(CFG, locate=lambda _job: (0.55, 0.10))
        job = PickJob(7, (0.55, 0.10), 0.0, CFG.quarantine_bin_m, 12.0, 0.0)
        arm.start(job)
        result = arm.abort("estop")
        assert result is not None and result.success is False and result.reason == "estop"
        assert arm.job is None

    def test_the_intercept_moves_downstream_with_the_belt(self):
        arm = Arm(CFG)
        still = PickJob(1, (0.45, 0.05), 0.0, CFG.quarantine_bin_m, 12.0, 0.0)
        moving = PickJob(1, (0.45, 0.05), 0.0, CFG.quarantine_bin_m, 12.0, 0.35)
        a = arm.solve_intercept(still)
        b = arm.solve_intercept(moving)
        assert a is not None and b is not None
        assert b[0] > a[0]

    def test_pick_cycle_time_is_plausible_for_a_sort_line(self):
        target = (0.55, 0.10)
        arm = Arm(CFG, locate=lambda _job: target)
        job = PickJob(1, target, 0.0, CFG.quarantine_bin_m, 12.0, 0.0, gt_uid=1)
        result, _t = self._run(arm, job)
        assert 0.5 < result.duration_s < 5.0
