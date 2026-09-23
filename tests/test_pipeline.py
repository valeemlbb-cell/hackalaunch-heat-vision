"""The closed loop, driven by a perfect detector so the rest is under test."""

from __future__ import annotations

import numpy as np
import pytest

from heatvision.config import DEFAULT
from heatvision.decide.policy import Action
from heatvision.pipeline import Station
from heatvision.sim.materials import BY_NAME
from heatvision.sim.scene import _instantiate
from tests.conftest import OracleDetector


def make_station(**kwargs) -> Station:
    detector = OracleDetector()
    station = Station(detector, DEFAULT, **kwargs)
    detector.station = station
    return station


def inject(station: Station, material: str, *, x: float = 0.25, y: float = 0.0):
    item = _instantiate(np.random.default_rng(5), BY_NAME[material], station.scene._next_uid, x, y)
    station.scene._next_uid += 1
    station.scene.items.append(item)
    return item


class TestFrameRecord:
    def test_a_step_returns_a_populated_record(self):
        station = make_station(seed=11)
        record = station.step()
        assert record.rgb.shape == (DEFAULT.belt.image_px, DEFAULT.belt.image_px, 3)
        assert record.thermal.shape[0] == DEFAULT.belt.image_px
        assert record.index == 0
        assert record.arm_state == "idle"
        assert len(record.decisions) == len(record.tracks)

    def test_time_and_index_advance(self):
        station = make_station(seed=12)
        records = [station.step() for _ in range(5)]
        assert [r.index for r in records] == [0, 1, 2, 3, 4]
        assert records[-1].time_s == pytest.approx(4 * station.dt)

    def test_tracks_accumulate_frames(self):
        station = make_station(seed=13, spawn_interval_s=999.0)
        for _ in range(4):
            record = station.step()
        assert any(t.frames_seen >= 3 for t in record.tracks)


class TestRemoval:
    def test_hazards_actually_get_removed(self):
        station = make_station(seed=21, hazard_rate=0.5)
        station.run(30.0)
        stats = station.stats.summary()
        assert stats["hazards_presented"] > 5
        assert stats["hazards_removed"] > 0
        assert stats["removal_rate"] > 0.4, stats

    def test_removed_items_leave_the_belt(self):
        station = make_station(seed=22, hazard_rate=0.6)
        station.run(20.0)
        removed = [it for it in station.scene.items if it.removed]
        assert removed
        assert all(it not in station.scene.visible() for it in removed)

    def test_the_counters_balance(self):
        station = make_station(seed=23)
        station.run(20.0)
        stats = station.stats.summary()
        assert stats["hazards_removed"] + stats["hazards_reached_crusher"] <= stats["hazards_presented"]
        assert stats["pick_attempts"] >= stats["hazards_removed"]

    def test_only_one_arm_job_runs_at_a_time(self):
        station = make_station(seed=24, hazard_rate=0.7)
        for _ in range(400):
            record = station.step()
            assert station.arm.job is None or station.arm.busy
            assert record.arm_state in {
                "idle", "to_intercept", "descend", "grip", "lift", "to_bin", "release", "return"
            }


class TestSafetyBehaviour:
    def test_a_venting_pouch_stops_the_belt(self):
        station = make_station(seed=31, hazard_rate=0.0, spawn_interval_s=999.0)
        item = inject(station, "pouch_swollen_venting", x=0.3)
        item.dt_core = 70.0
        item.rise_rate = 5.0
        for _ in range(12):
            record = station.step()
            if record.estop:
                break
        assert record.estop is True
        assert record.belt_speed == 0.0
        assert station.stats.estops >= 1

    def test_the_arm_never_grips_a_venting_cell(self):
        """A venting cell leaves on a human's shovel, never in the gripper."""
        station = make_station(seed=32, hazard_rate=0.0, spawn_interval_s=999.0)
        item = inject(station, "pouch_swollen_venting", x=0.3)
        item.dt_core = 80.0
        item.rise_rate = 6.0
        for _ in range(60):
            record = station.step()
            for decision in record.decisions:
                if decision.action is Action.EMERGENCY_STOP:
                    assert decision.grip_force_n == 0.0
        assert station.stats.hazards_removed == 0, "the robot must not pick a venting cell"
        assert station.stats.operator_removals >= 1
        assert station.stats.estops >= 1

    def test_the_line_restarts_after_the_operator_clears_the_stop(self):
        station = make_station(seed=33, hazard_rate=0.0, spawn_interval_s=999.0, estop_clear_s=0.6)
        item = inject(station, "pouch_swollen_venting", x=0.3)
        item.dt_core = 75.0
        item.rise_rate = 5.0
        speeds = [station.step().belt_speed for _ in range(60)]
        assert 0.0 in speeds
        assert speeds[-1] == pytest.approx(DEFAULT.belt.speed_mps)

    def test_a_hot_but_stable_pouch_goes_to_the_quench_bin(self):
        station = make_station(seed=34, hazard_rate=0.0, spawn_interval_s=999.0)
        item = inject(station, "pouch_lipo", x=0.25)
        item.dt_core = 30.0
        item.emissivity = 0.92
        actions = set()
        for _ in range(80):
            record = station.step()
            actions.update(d.action for d in record.decisions)
            if station.arm.job is not None:
                assert station.arm.job.bin_xy == DEFAULT.arm.quench_bin_m
                assert station.arm.job.grip_force_n == DEFAULT.policy.swollen_force_limit_n
                break
        assert Action.EXTRACT_QUENCH in actions

    def test_the_belt_stops_for_a_quench_pick_and_restarts_after(self):
        station = make_station(seed=35, hazard_rate=0.0, spawn_interval_s=999.0)
        item = inject(station, "pouch_lipo", x=0.25)
        item.dt_core = 30.0
        item.emissivity = 0.92
        speeds = [station.step().belt_speed for _ in range(140)]
        assert min(speeds) == 0.0
        assert speeds[-1] == pytest.approx(DEFAULT.belt.speed_mps)

    def test_items_past_the_pick_window_are_flagged_not_chased(self):
        station = make_station(seed=36, hazard_rate=0.0, spawn_interval_s=999.0)
        inject(station, "cell_18650_warm", x=1.08)
        seen = set()
        for _ in range(12):
            record = station.step()
            seen.update(d.action for d in record.decisions)
        assert Action.TOO_LATE in seen
        assert station.stats.too_late >= 1


class TestLogging:
    def test_every_action_is_written_to_the_log(self):
        station = make_station(seed=41, hazard_rate=0.6)
        station.run(15.0)
        assert station.log
        kinds = {row["event"] for row in station.log}
        assert "pick_start" in kinds
        assert "pick_result" in kinds
        for row in station.log:
            assert "t" in row

    def test_pick_results_record_the_grasp_error(self):
        station = make_station(seed=42, hazard_rate=0.6)
        station.run(20.0)
        results = [r for r in station.log if r["event"] == "pick_result" and r["success"]]
        assert results
        assert all(r["grasp_error_mm"] is not None for r in results)

    def test_events_are_surfaced_on_the_frame_record(self):
        station = make_station(seed=43, hazard_rate=0.7)
        events = []
        for _ in range(240):
            events.extend(station.step().events)
        assert any(e.startswith("PICK ") for e in events)


class TestSummary:
    def test_summary_has_the_reported_fields(self):
        station = make_station(seed=51)
        station.run(8.0)
        summary = station.stats.summary()
        for key in (
            "frames",
            "hazards_presented",
            "hazards_removed",
            "removal_rate",
            "pick_success_rate",
            "false_positive_picks",
            "emergency_stops",
            "mean_detect_ms",
        ):
            assert key in summary

    def test_rates_are_fractions(self):
        station = make_station(seed=52)
        station.run(12.0)
        summary = station.stats.summary()
        assert 0.0 <= summary["removal_rate"] <= 1.0
        assert 0.0 <= (summary["pick_success_rate"] or 0.0) <= 1.0
