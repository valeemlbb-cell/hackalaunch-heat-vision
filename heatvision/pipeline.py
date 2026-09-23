"""The closed loop: sense -> track -> decide -> plan -> act -> verify.

One :meth:`Station.step` is one control cycle of the whole cell. Nothing in
here knows what the "right" answer is: the detector output drives the tracker,
the tracker drives the policy, the policy drives the arm, and the arm either
gets the object or does not. Ground truth is touched in exactly two places,
both marked ``SIMULATION ONLY``:

* attaching a ground-truth uid to a track, so the simulated physics knows
  which object the jaws closed on, and so the scorer can tell a real removal
  from a lucky one;
* the ``locate`` callback, which stands in for the real world.

The detector and the policy never see either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arm.kinematics import grasp_angle
from .arm.robot import Arm, ArmState, PickJob, PickResult
from .config import DEFAULT, CellConfig
from .decide.policy import Action, Decision, decide_all
from .decide.thermal import ThermalReading, belt_baseline, measure
from .decide.tracker import BeltTracker, Track
from .detect.decode import Detection, iou_matrix
from .detect.infer import Detector
from .sim.render import item_box_px, px_to_m, render_frame
from .sim.scene import Item, Scene


@dataclass
class FrameRecord:
    """Everything that happened in one control cycle."""

    index: int
    time_s: float
    rgb: np.ndarray
    thermal: np.ndarray
    detections: list[Detection]
    tracks: list[Track]
    decisions: list[Decision]
    belt_speed: float
    arm_state: str
    arm_tool_xy: tuple[float, float]
    arm_joints: tuple[float, float, float, float]
    holding: bool
    estop: bool
    events: list[str] = field(default_factory=list)
    baseline_dt: float = 0.0


@dataclass
class RunStats:
    frames: int = 0
    hazards_seen: int = 0
    hazards_removed: int = 0
    #: venting items lifted out by a human after an emergency stop
    operator_removals: int = 0
    hazards_crushed: int = 0
    pick_attempts: int = 0
    pick_success: int = 0
    false_positive_picks: int = 0
    estops: int = 0
    operator_alerts: int = 0
    too_late: int = 0
    cycle_times: list[float] = field(default_factory=list)
    detect_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        contained = self.hazards_removed + self.operator_removals
        removal = self.hazards_removed / self.hazards_seen if self.hazards_seen else 0.0
        containment = contained / self.hazards_seen if self.hazards_seen else 0.0
        pick = self.pick_success / self.pick_attempts if self.pick_attempts else 0.0
        return {
            "frames": self.frames,
            "hazards_presented": self.hazards_seen,
            "hazards_removed": self.hazards_removed,
            "hazards_operator_removed": self.operator_removals,
            "hazards_reached_crusher": self.hazards_crushed,
            "removal_rate": round(removal, 4),
            "containment_rate": round(containment, 4),
            "pick_attempts": self.pick_attempts,
            "pick_success_rate": round(pick, 4),
            "false_positive_picks": self.false_positive_picks,
            "emergency_stops": self.estops,
            "operator_alerts": self.operator_alerts,
            "too_late_flags": self.too_late,
            "mean_cycle_s": round(float(np.mean(self.cycle_times)), 3) if self.cycle_times else None,
            "mean_detect_ms": round(float(np.mean(self.detect_ms)), 2) if self.detect_ms else None,
        }


class Station:
    """A complete sorting station running against the simulator."""

    def __init__(
        self,
        detector: Detector,
        cfg: CellConfig = DEFAULT,
        *,
        seed: int = 7,
        fps: float = 20.0,
        hazard_rate: float = 0.28,
        spawn_interval_s: float = 0.75,
        estop_clear_s: float = 2.5,
        initial_items: tuple[int, int] = (4, 8),
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / fps
        self.hazard_rate = hazard_rate
        self.spawn_interval_s = spawn_interval_s
        self.estop_clear_s = estop_clear_s

        self.scene = Scene.populate(self.rng, cfg.belt, n_items=initial_items, hazard_rate=hazard_rate)
        self.tracker = BeltTracker(cfg.belt.speed_mps)
        self.arm = Arm(cfg.arm, locate=self._locate, on_grasp=self._on_grasp)
        self.belt_speed = cfg.belt.speed_mps
        self.estop = False
        self._estop_until = 0.0
        self._estop_uid: int | None = None
        self.time_s = 0.0
        self.index = 0
        self._next_spawn = 0.0
        self.stats = RunStats()
        self.log: list[dict] = []
        self._seen_hazards: set[int] = set()
        self._claimed_uids: set[int] = set()

    # ------------------------------------------------------- world hooks
    def _item_by_uid(self, uid: int | None) -> Item | None:
        if uid is None:
            return None
        for item in self.scene.items:
            if item.uid == uid:
                return item
        return None

    def _locate(self, job: PickJob) -> tuple[float, float] | None:
        """SIMULATION ONLY: where the physical object actually is right now."""
        item = self._item_by_uid(job.gt_uid)
        if item is None or item.removed or item.crushed:
            return None
        return (item.x_m, item.y_m)

    def _on_grasp(self, job: PickJob) -> None:
        """SIMULATION ONLY: the jaws closed, so take the item off the belt."""
        item = self._item_by_uid(job.gt_uid)
        if item is not None:
            item.removed = True

    # ------------------------------------------------------------ sensing
    def _detect(self, rgb: np.ndarray, thermal: np.ndarray) -> tuple[list[Detection], list[ThermalReading], list[tuple[float, float]], float]:
        import time as _time

        t0 = _time.perf_counter()
        detections = self.detector.detect(rgb, thermal)
        self.stats.detect_ms.append((_time.perf_counter() - t0) * 1000.0)
        base = belt_baseline(thermal)
        readings = [measure(thermal, d.box, baseline=base) for d in detections]
        centers = [px_to_m(self.cfg.belt, *d.center) for d in detections]
        return detections, readings, centers, base

    def _attach_ground_truth(self, tracks: list[Track]) -> None:
        """SIMULATION ONLY: associate tracks with real items by box overlap.

        Used for the physics of grasping and for scoring. The detector, the
        tracker and the policy are all unaffected by this.
        """
        candidates = [it for it in self.scene.visible()]
        if not candidates:
            return
        gt_boxes = np.array([item_box_px(self.cfg.belt, it) for it in candidates], np.float32)
        pending = [t for t in tracks if t.gt_uid is None]
        if not pending:
            return
        tr_boxes = np.array([t.box for t in pending], np.float32)
        ious = iou_matrix(tr_boxes, gt_boxes)
        for i, track in enumerate(pending):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= 0.25:
                track.gt_uid = candidates[j].uid

    # ----------------------------------------------------------- dispatch
    def _dispatch(self, decisions: list[Decision], events: list[str]) -> None:
        for decision in decisions:
            track = self.tracker.get(decision.track_id)
            if track is None:
                continue
            if decision.action is Action.EMERGENCY_STOP and not self.estop:
                self.estop = True
                self._estop_until = self.time_s + self.estop_clear_s
                self._estop_uid = track.gt_uid
                self.belt_speed = 0.0
                self.stats.estops += 1
                if self.arm.busy:
                    self.arm.abort("estop")
                events.append(f"E-STOP track {track.track_id}: {decision.reason}")
                self._record(decision, "estop")
                return
            if decision.action is Action.ALERT_OPERATOR and not getattr(track, "_alerted", False):
                track._alerted = True  # type: ignore[attr-defined]
                self.stats.operator_alerts += 1
                events.append(f"ALERT track {track.track_id}: {decision.reason}")
                self._record(decision, "alert")
            if decision.action is Action.TOO_LATE and not getattr(track, "_flagged", False):
                track._flagged = True  # type: ignore[attr-defined]
                self.stats.too_late += 1
                events.append(f"TOO LATE track {track.track_id}: {decision.reason}")
                self._record(decision, "too_late")

        if self.estop or self.arm.busy:
            return

        for decision in decisions:
            if not decision.is_extraction:
                continue
            track = self.tracker.get(decision.track_id)
            if track is None or track.claimed or track.removed:
                continue
            if track.gt_uid is not None and track.gt_uid in self._claimed_uids:
                continue
            if self._launch(track, decision, events):
                return

    def _launch(self, track: Track, decision: Decision, events: list[str]) -> bool:
        item = self._item_by_uid(track.gt_uid)
        tool_angle = grasp_angle(item.theta) if item is not None else 0.0
        stop_belt = decision.action is Action.EXTRACT_QUENCH
        if stop_belt:
            self.belt_speed = 0.0
        job = PickJob(
            track_id=track.track_id,
            target_xy=track.center_m,
            tool_angle=tool_angle,
            bin_xy=self.cfg.arm.quench_bin_m if stop_belt else self.cfg.arm.quarantine_bin_m,
            grip_force_n=decision.grip_force_n,
            belt_speed_mps=self.belt_speed,
            label=decision.cls_name,
            gt_uid=track.gt_uid,
        )
        if not self.arm.start(job):
            if stop_belt:
                self.belt_speed = self.cfg.belt.speed_mps
            return False
        track.claimed = True
        if track.gt_uid is not None:
            self._claimed_uids.add(track.gt_uid)
        self.stats.pick_attempts += 1
        events.append(
            f"PICK track {track.track_id} ({decision.cls_name}) -> "
            f"{'quench' if stop_belt else 'quarantine'} @ {decision.grip_force_n:.0f} N"
        )
        self._record(decision, "pick_start")
        return True

    def _record(self, decision: Decision, event: str) -> None:
        row = decision.to_dict()
        row["t"] = round(self.time_s, 3)
        row["event"] = event
        self.log.append(row)

    def _on_pick_result(self, result: PickResult, events: list[str]) -> None:
        track = self.tracker.get(result.track_id)
        if result.success:
            self.stats.pick_success += 1
            self.stats.cycle_times.append(result.duration_s)
            item = self._item_by_uid(result.gt_uid)
            if item is None or not item.is_hazard:
                self.stats.false_positive_picks += 1
                events.append(f"PICKED NON-HAZARD track {result.track_id} (false positive)")
            else:
                self.stats.hazards_removed += 1
                events.append(f"REMOVED track {result.track_id} in {result.duration_s:.2f}s")
            if track is not None:
                track.removed = True
                self.tracker.drop(track.track_id)
        else:
            if result.reason == "target_gone" and result.gt_uid is None:
                self.stats.false_positive_picks += 1
            events.append(
                f"PICK FAILED track {result.track_id}: {result.reason}"
                + (f" (err {result.grasp_error_m * 1000:.0f} mm)" if result.grasp_error_m else "")
            )
            if track is not None:
                track.claimed = False
            if result.gt_uid is not None:
                self._claimed_uids.discard(result.gt_uid)
        self.log.append(
            {
                "t": round(self.time_s, 3),
                "event": "pick_result",
                "track_id": result.track_id,
                "success": result.success,
                "reason": result.reason,
                "duration_s": round(result.duration_s, 3),
                "grasp_error_mm": round(result.grasp_error_m * 1000, 1) if result.grasp_error_m else None,
            }
        )

    # --------------------------------------------------------------- step
    def step(self) -> FrameRecord:
        events: list[str] = []

        # 1. world
        self.scene.advance(self.dt, speed_mps=self.belt_speed)
        if self.time_s >= self._next_spawn and self.belt_speed > 0.0:
            self.scene.spawn(self.rng, hazard_rate=self.hazard_rate)
            self._next_spawn = self.time_s + self.spawn_interval_s

        # 2. e-stop recovery. A venting cell is never handled by the robot: a
        #    human in PPE lifts it into a quench drum, acknowledges the stop and
        #    restarts the line. That manual extraction is counted separately
        #    from the robot's own removals so the two never get conflated.
        if self.estop and self.time_s >= self._estop_until:
            item = self._item_by_uid(self._estop_uid)
            if item is not None and not item.removed:
                item.removed = True
                if item.is_hazard:
                    self.stats.operator_removals += 1
                events.append("operator extracted the venting item into a quench drum")
            self.estop = False
            self._estop_uid = None
            self.belt_speed = self.cfg.belt.speed_mps
            events.append("E-STOP cleared by operator, belt restarted")

        # 3. book-keeping on hazards that entered / left the view
        for item in self.scene.items:
            if item.is_hazard and item.uid not in self._seen_hazards and item.x_m >= 0.0:
                self._seen_hazards.add(item.uid)
                self.stats.hazards_seen += 1
            if item.is_hazard and item.crushed and not item.removed and not getattr(item, "_counted", False):
                item._counted = True  # type: ignore[attr-defined]
                self.stats.hazards_crushed += 1

        # 4. sense
        rgb, thermal = render_frame(self.rng, self.scene, self.cfg.thermal)
        detections, readings, centers, base = self._detect(rgb, thermal)

        # 5. track + decide
        tracks = self.tracker.update(detections, readings, centers, self.time_s, self.dt)
        self._attach_ground_truth(tracks)
        reachable = {
            t.track_id: self.arm.can_reach(*t.center_m) or self.arm.busy for t in tracks
        }
        decisions = decide_all(tracks, self.cfg.policy, reachable=reachable)

        # 6. act
        self._dispatch(decisions, events)
        result = self.arm.step(self.dt)
        if result is not None:
            self._on_pick_result(result, events)
        if self.arm.state is ArmState.IDLE and not self.estop and self.belt_speed == 0.0:
            self.belt_speed = self.cfg.belt.speed_mps
            events.append("belt restarted after quench pick")

        record = FrameRecord(
            index=self.index,
            time_s=self.time_s,
            rgb=rgb,
            thermal=thermal,
            detections=detections,
            tracks=tracks,
            decisions=decisions,
            belt_speed=self.belt_speed,
            arm_state=self.arm.state.value,
            arm_tool_xy=self.arm.tool_xy,
            arm_joints=self.arm.joints.as_tuple(),
            holding=self.arm.holding,
            estop=self.estop,
            events=events,
            baseline_dt=base,
        )
        self.stats.frames += 1
        self.index += 1
        self.time_s += self.dt
        return record

    def run(self, seconds: float) -> list[FrameRecord]:
        return [self.step() for _ in range(int(round(seconds / self.dt)))]

    def hazards_removed(self) -> int:
        return sum(1 for it in self.scene.items if it.is_hazard and it.removed)
