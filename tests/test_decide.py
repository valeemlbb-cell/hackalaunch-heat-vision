"""Thermal measurement, tracking and the safety policy."""

from __future__ import annotations

import numpy as np
import pytest

from heatvision.config import CLASS_INDEX, PolicyConfig
from heatvision.decide.policy import Action, HazardTier, classify, decide, decide_all, grip_force_for
from heatvision.decide.thermal import ThermalReading, belt_baseline, measure, rise_rate
from heatvision.decide.tracker import BeltTracker, Track
from heatvision.detect.decode import Detection

POLICY = PolicyConfig()


def make_track(
    *,
    score: float = 0.9,
    peak_dt: float = 8.0,
    cls: str = "cell",
    frames: int = 5,
    x_m: float = 0.5,
    history: list[tuple[float, float]] | None = None,
) -> Track:
    reading = ThermalReading(peak_dt=peak_dt, mean_dt=peak_dt * 0.7, hot_area_frac=0.3, baseline_dt=0.6)
    return Track(
        track_id=1,
        cls_id=CLASS_INDEX[cls],
        box=(10.0, 10.0, 30.0, 26.0),
        score=score,
        center_m=(x_m, 0.0),
        reading=reading,
        first_seen_s=0.0,
        last_seen_s=frames * 0.05,
        frames_seen=frames,
        thermal_history=history if history is not None else [(0.0, peak_dt), (0.1, peak_dt)],
    )


class TestThermalMeasurement:
    def test_baseline_is_immune_to_a_few_hot_pixels(self):
        frame = np.full((64, 64), 0.5, np.float32)
        frame[:4, :4] = 90.0
        assert belt_baseline(frame) == pytest.approx(0.5, abs=1e-5)

    def test_measure_reports_the_rise_above_the_belt(self):
        frame = np.full((64, 64), 1.0, np.float32)
        frame[20:40, 20:40] = 26.0
        reading = measure(frame, (20, 20, 40, 40))
        assert reading.peak_dt == pytest.approx(25.0, abs=0.5)
        assert reading.hot_area_frac > 0.9
        assert reading.is_warm

    def test_measure_on_a_cold_object(self):
        frame = np.full((64, 64), 1.0, np.float32)
        reading = measure(frame, (20, 20, 40, 40))
        assert reading.peak_dt == pytest.approx(0.0, abs=0.2)
        assert not reading.is_warm

    def test_measure_clamps_a_box_that_runs_off_frame(self):
        frame = np.full((32, 32), 2.0, np.float32)
        reading = measure(frame, (-50, -50, 5, 5))
        assert np.isfinite(reading.peak_dt)

    def test_measure_on_a_degenerate_box_is_safe(self):
        frame = np.full((32, 32), 2.0, np.float32)
        reading = measure(frame, (100, 100, 101, 101))
        assert np.isfinite(reading.peak_dt)


class TestRiseRate:
    def test_a_steady_reading_has_no_slope(self):
        assert rise_rate([(0.0, 10.0), (0.5, 10.0), (1.0, 10.0)]) == pytest.approx(0.0, abs=1e-6)

    def test_a_linear_rise_recovers_its_slope(self):
        history = [(t / 10, 5.0 + 4.0 * (t / 10)) for t in range(11)]
        assert rise_rate(history) == pytest.approx(4.0, abs=1e-6)

    def test_too_few_points_returns_zero(self):
        assert rise_rate([(0.0, 5.0), (0.1, 30.0)]) == 0.0

    def test_cooling_gives_a_negative_slope(self):
        history = [(t / 10, 40.0 - 3.0 * (t / 10)) for t in range(8)]
        assert rise_rate(history) < -2.0


class TestTracker:
    def _det(self, cls_id: int = 0, score: float = 0.8) -> Detection:
        return Detection(box=(10.0, 10.0, 30.0, 26.0), score=score, cls_id=cls_id)

    def test_first_sighting_creates_a_track(self):
        tracker = BeltTracker(0.35)
        reading = ThermalReading(8.0, 5.0, 0.2, 0.5)
        tracks = tracker.update([self._det()], [reading], [(0.4, 0.0)], 0.0, 0.05)
        assert len(tracks) == 1
        assert tracks[0].frames_seen == 1

    def test_a_moving_item_keeps_its_track_id(self):
        tracker = BeltTracker(0.35)
        reading = ThermalReading(8.0, 5.0, 0.2, 0.5)
        ids = []
        x = 0.30
        for i in range(6):
            tracks = tracker.update([self._det()], [reading], [(x, 0.0)], i * 0.05, 0.05)
            ids.append(tracks[0].track_id)
            x += 0.35 * 0.05
        assert len(set(ids)) == 1
        assert tracker.tracks[ids[0]].frames_seen == 6

    def test_a_distant_detection_starts_a_new_track(self):
        tracker = BeltTracker(0.35)
        reading = ThermalReading(8.0, 5.0, 0.2, 0.5)
        tracker.update([self._det()], [reading], [(0.2, 0.0)], 0.0, 0.05)
        tracks = tracker.update([self._det()], [reading], [(0.9, 0.3)], 0.05, 0.05)
        assert tracks[0].track_id == 1

    def test_a_track_is_dropped_after_repeated_misses(self):
        tracker = BeltTracker(0.35, max_misses=2)
        reading = ThermalReading(8.0, 5.0, 0.2, 0.5)
        tracker.update([self._det()], [reading], [(0.3, 0.0)], 0.0, 0.05)
        for i in range(1, 5):
            tracker.update([], [], [], i * 0.05, 0.05)
        assert tracker.tracks == {}

    def test_thermal_history_feeds_the_rise_rate(self):
        tracker = BeltTracker(0.0)
        for i in range(6):
            reading = ThermalReading(10.0 + 5.0 * i * 0.1, 5.0, 0.2, 0.5)
            tracks = tracker.update([self._det()], [reading], [(0.4, 0.0)], i * 0.1, 0.1)
        assert tracks[0].rise_rate_k_s == pytest.approx(5.0, abs=0.5)


class TestClassify:
    def test_no_vision_evidence_is_never_a_hazard(self):
        assert classify(make_track(score=0.05, peak_dt=60.0), POLICY) is HazardTier.NONE

    def test_strong_vision_alone_is_confirmed(self):
        assert classify(make_track(score=0.85, peak_dt=0.5), POLICY) is HazardTier.CONFIRMED

    def test_weak_vision_plus_heat_is_promoted_to_confirmed(self):
        assert classify(make_track(score=0.38, peak_dt=9.0), POLICY) is HazardTier.CONFIRMED

    def test_weak_vision_without_heat_stays_suspect(self):
        assert classify(make_track(score=0.38, peak_dt=1.0), POLICY) is HazardTier.SUSPECT

    def test_hot_confirmed_item_is_a_thermal_event(self):
        assert classify(make_track(score=0.8, peak_dt=30.0), POLICY) is HazardTier.THERMAL_EVENT

    def test_very_hot_item_is_runaway(self):
        assert classify(make_track(score=0.8, peak_dt=55.0), POLICY) is HazardTier.RUNAWAY

    def test_fast_rise_is_runaway_even_below_the_absolute_threshold(self):
        history = [(t / 10, 20.0 + 5.0 * (t / 10)) for t in range(8)]
        track = make_track(score=0.8, peak_dt=27.0, history=history)
        assert classify(track, POLICY) is HazardTier.RUNAWAY


class TestPolicyRules:
    def test_R0_hot_decoy_alerts_instead_of_stopping_the_line(self):
        """The headline safety property: no nuisance stops on hot scrap."""
        decision = decide(make_track(score=0.08, peak_dt=48.0), POLICY)
        assert decision.action is Action.ALERT_OPERATOR
        assert decision.rule == "R0"
        assert decision.grip_force_n == 0.0

    def test_R1_cold_clutter_is_ignored(self):
        decision = decide(make_track(score=0.05, peak_dt=0.4), POLICY)
        assert decision.action is Action.IGNORE
        assert decision.rule == "R1"

    def test_R2_runaway_stops_the_belt_and_never_grips(self):
        decision = decide(make_track(score=0.8, peak_dt=60.0), POLICY)
        assert decision.action is Action.EMERGENCY_STOP
        assert decision.grip_force_n == 0.0
        assert decision.tier is HazardTier.RUNAWAY

    def test_R3_a_single_frame_is_not_enough_to_move_the_arm(self):
        decision = decide(make_track(score=0.9, peak_dt=8.0, frames=1), POLICY)
        assert decision.action is Action.OBSERVE
        assert decision.rule == "R3"

    def test_R4_past_the_pick_window_is_flagged_not_attempted(self):
        decision = decide(make_track(score=0.9, peak_dt=8.0, x_m=1.10), POLICY)
        assert decision.action is Action.TOO_LATE

    def test_R4_unreachable_targets_are_flagged(self):
        decision = decide(make_track(score=0.9, peak_dt=8.0), POLICY, reachable=False)
        assert decision.action is Action.TOO_LATE

    def test_R5_hot_but_stable_goes_to_the_quench_bin_gently(self):
        decision = decide(make_track(score=0.9, peak_dt=30.0), POLICY)
        assert decision.action is Action.EXTRACT_QUENCH
        assert decision.grip_force_n == POLICY.swollen_force_limit_n

    def test_R6_confirmed_cell_is_extracted_to_quarantine(self):
        decision = decide(make_track(score=0.9, peak_dt=7.0), POLICY)
        assert decision.action is Action.EXTRACT_QUARANTINE
        assert decision.grip_force_n == POLICY.grip_force_limit_n

    def test_R7_weak_and_cold_keeps_watching(self):
        decision = decide(make_track(score=0.38, peak_dt=1.0), POLICY)
        assert decision.action is Action.OBSERVE
        assert decision.rule == "R7"

    def test_every_decision_carries_its_numbers(self):
        payload = decide(make_track(), POLICY).to_dict()
        for key in ("rule", "action", "vision_score", "peak_dt_k", "rise_rate_k_s", "reason"):
            assert key in payload


class TestGripForce:
    def test_pouches_always_get_the_gentle_limit(self):
        track = make_track(cls="pouch", peak_dt=7.0)
        assert grip_force_for(track, HazardTier.CONFIRMED, POLICY) == POLICY.swollen_force_limit_n

    def test_a_cold_cell_gets_the_normal_limit(self):
        track = make_track(cls="cell", peak_dt=7.0)
        assert grip_force_for(track, HazardTier.CONFIRMED, POLICY) == POLICY.grip_force_limit_n

    def test_the_gentle_limit_is_below_the_normal_one(self):
        assert POLICY.swollen_force_limit_n < POLICY.grip_force_limit_n


class TestPriority:
    def test_hotter_hazards_are_handled_first(self):
        cold = make_track(score=0.9, peak_dt=7.0)
        hot = make_track(score=0.9, peak_dt=30.0)
        hot.track_id = 2
        decisions = decide_all([cold, hot], POLICY)
        assert decisions[0].track_id == 2

    def test_decide_all_returns_one_decision_per_track(self):
        tracks = [make_track(), make_track()]
        tracks[1].track_id = 2
        assert len(decide_all(tracks, POLICY)) == 2
