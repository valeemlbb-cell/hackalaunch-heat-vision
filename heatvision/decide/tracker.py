"""Frame-to-frame association on a moving belt.

Tracking is not a nicety here, it is a safety requirement:

* a single-frame detection is not enough evidence to put a robot arm into the
  stream, so the policy demands ``min_track_frames`` consistent sightings;
* heat *rate* of change is the signal that separates a venting cell from a
  merely warm object, and a rate needs a history;
* the arm has to intercept a moving target, so we need a velocity estimate.

Association is nearest-neighbour in predicted-position space, which is
sufficient because the belt motion is known and objects are sparse relative to
the frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..detect.decode import Detection
from .thermal import ThermalReading, rise_rate


@dataclass
class Track:
    track_id: int
    cls_id: int
    box: tuple[float, float, float, float]
    score: float
    center_m: tuple[float, float]
    reading: ThermalReading
    first_seen_s: float
    last_seen_s: float
    frames_seen: int = 1
    misses: int = 0
    #: ``(time_s, peak_dt)`` samples used for the rise-rate fit
    thermal_history: list[tuple[float, float]] = field(default_factory=list)
    #: set once the pipeline has committed to picking this track
    claimed: bool = False
    #: set once the arm has actually taken it off the belt
    removed: bool = False
    #: ground-truth item uid, populated only in simulation for scoring
    gt_uid: int | None = None

    @property
    def age_s(self) -> float:
        return self.last_seen_s - self.first_seen_s

    @property
    def rise_rate_k_s(self) -> float:
        return rise_rate(self.thermal_history)

    @property
    def peak_dt(self) -> float:
        return self.reading.peak_dt

    def velocity_mps(self, belt_speed: float) -> tuple[float, float]:
        """Items move with the belt; cross-belt drift is negligible."""
        return (belt_speed, 0.0)


class BeltTracker:
    """Nearest-neighbour tracker with belt-motion prediction."""

    def __init__(self, belt_speed_mps: float, *, max_match_m: float = 0.09, max_misses: int = 3) -> None:
        self.belt_speed = belt_speed_mps
        self.max_match_m = max_match_m
        self.max_misses = max_misses
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def _predict(self, track: Track, dt: float) -> tuple[float, float]:
        x, y = track.center_m
        return (x + self.belt_speed * dt, y)

    def update(
        self,
        detections: list[Detection],
        readings: list[ThermalReading],
        centers_m: list[tuple[float, float]],
        time_s: float,
        dt: float,
    ) -> list[Track]:
        """Associate this frame's detections and return the live tracks."""
        predicted = {tid: self._predict(t, dt) for tid, t in self.tracks.items()}
        unmatched = set(self.tracks)
        matched_now: set[int] = set()

        order = sorted(range(len(detections)), key=lambda i: -detections[i].score)
        for i in order:
            det, reading, center = detections[i], readings[i], centers_m[i]
            best_id, best_dist = None, self.max_match_m
            for tid in unmatched:
                px, py = predicted[tid]
                dist = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = tid, dist
            if best_id is None:
                track = Track(
                    track_id=self._next_id,
                    cls_id=det.cls_id,
                    box=det.box,
                    score=det.score,
                    center_m=center,
                    reading=reading,
                    first_seen_s=time_s,
                    last_seen_s=time_s,
                )
                track.thermal_history.append((time_s, reading.peak_dt))
                self.tracks[self._next_id] = track
                matched_now.add(self._next_id)
                self._next_id += 1
            else:
                track = self.tracks[best_id]
                track.cls_id = det.cls_id if det.score > track.score else track.cls_id
                track.box = det.box
                track.score = max(det.score, 0.7 * track.score + 0.3 * det.score)
                track.center_m = center
                track.reading = reading
                track.last_seen_s = time_s
                track.frames_seen += 1
                track.misses = 0
                track.thermal_history.append((time_s, reading.peak_dt))
                if len(track.thermal_history) > 40:
                    del track.thermal_history[0]
                unmatched.discard(best_id)
                matched_now.add(best_id)

        for tid in list(unmatched):
            track = self.tracks[tid]
            track.misses += 1
            # keep coasting it along the belt so a one-frame dropout does not
            # spawn a duplicate track next frame
            track.center_m = self._predict(track, dt)
            if track.misses > self.max_misses and not track.claimed:
                del self.tracks[tid]

        return [self.tracks[tid] for tid in sorted(matched_now) if tid in self.tracks]

    def get(self, track_id: int) -> Track | None:
        return self.tracks.get(track_id)

    def drop(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)
