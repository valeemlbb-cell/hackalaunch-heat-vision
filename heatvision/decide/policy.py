"""Safety policy: what to do about a tracked object.

Design rules, in order of authority:

1. **Neither modality may overrule the other silently.** Vision decides *what*
   it is, thermal decides *how dangerous it is right now*, and the two are
   combined by explicit rules that a plant engineer can read.
2. **No unsupported line stops.** A 50 K object with no visual evidence of a
   battery is a hot brake disc, not a runaway cell; it raises an operator
   alert, it does not halt production. Nuisance stops are how safety systems
   get switched off.
3. **Never grip something that is venting.** Above the runaway threshold the
   correct action is to stop the belt and call a human, not to squeeze a cell
   that is already failing.
4. **Every decision is explainable.** Each one carries the rule id and the
   numbers that fired it, and the pipeline logs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import CLASS_NAMES, PolicyConfig
from .tracker import Track


class Action(str, Enum):
    IGNORE = "ignore"  # not a lithium item
    OBSERVE = "observe"  # not enough evidence yet, keep watching
    EXTRACT_QUARANTINE = "extract_quarantine"  # normal pick to the battery bin
    EXTRACT_QUENCH = "extract_quench"  # hot but stable: gentle pick to the sand bin
    EMERGENCY_STOP = "emergency_stop"  # venting cell: halt belt, call operator
    ALERT_OPERATOR = "alert_operator"  # hot object with no lithium evidence
    TOO_LATE = "too_late"  # past the pick window, flag downstream


#: which actions actually put the arm into the stream
EXTRACTION_ACTIONS = frozenset({Action.EXTRACT_QUARANTINE, Action.EXTRACT_QUENCH})


class HazardTier(str, Enum):
    NONE = "none"
    SUSPECT = "suspect"
    CONFIRMED = "confirmed"
    THERMAL_EVENT = "thermal_event"
    RUNAWAY = "runaway"


@dataclass(frozen=True)
class Decision:
    action: Action
    tier: HazardTier
    rule: str
    reason: str
    grip_force_n: float
    priority: float
    track_id: int
    cls_name: str
    vision_score: float
    peak_dt: float
    rise_rate_k_s: float

    @property
    def is_extraction(self) -> bool:
        return self.action in EXTRACTION_ACTIONS

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "action": self.action.value,
            "tier": self.tier.value,
            "rule": self.rule,
            "reason": self.reason,
            "class": self.cls_name,
            "vision_score": round(self.vision_score, 3),
            "peak_dt_k": round(self.peak_dt, 2),
            "rise_rate_k_s": round(self.rise_rate_k_s, 2),
            "grip_force_n": round(self.grip_force_n, 2),
            "priority": round(self.priority, 3),
        }


def _priority(track: Track, tier: HazardTier, cfg: PolicyConfig) -> float:
    """Higher is more urgent: hot things first, then whatever is about to fall
    into the crusher."""
    tier_weight = {
        HazardTier.RUNAWAY: 100.0,
        HazardTier.THERMAL_EVENT: 60.0,
        HazardTier.CONFIRMED: 20.0,
        HazardTier.SUSPECT: 5.0,
        HazardTier.NONE: 0.0,
    }[tier]
    urgency = track.center_m[0] / max(cfg.max_reach_x_m, 1e-3)  # closer to the crusher = sooner
    return tier_weight + 10.0 * min(urgency, 1.5) + track.score


def classify(track: Track, cfg: PolicyConfig) -> HazardTier:
    """Fuse the two modalities into a hazard tier."""
    vision_ok = track.score >= cfg.vision_accept
    vision_weak = track.score >= cfg.vision_review
    hot = track.peak_dt >= cfg.dt_hot
    warm = track.peak_dt >= cfg.dt_warm
    # A rate trigger on its own is not enough: sensor noise can fake a slope on
    # a cold object, so a fast rise only counts once the item is already hot.
    runaway = track.peak_dt >= cfg.dt_runaway or (
        track.peak_dt >= cfg.dt_hot and track.rise_rate_k_s >= cfg.rise_rate_runaway
    )

    if not vision_weak:
        # thermal alone is never enough to call something a battery
        return HazardTier.NONE
    if runaway and vision_weak:
        return HazardTier.RUNAWAY
    if hot and vision_weak:
        return HazardTier.THERMAL_EVENT
    if vision_ok:
        return HazardTier.CONFIRMED
    if vision_weak and warm:
        # weak visual hit corroborated by self-heating
        return HazardTier.CONFIRMED
    return HazardTier.SUSPECT


def grip_force_for(track: Track, tier: HazardTier, cfg: PolicyConfig) -> float:
    """Pouches and anything warm get the gentle limit: puncture starts fires."""
    gentle = CLASS_NAMES[track.cls_id] == "pouch" or tier is HazardTier.THERMAL_EVENT
    return cfg.swollen_force_limit_n if gentle else cfg.grip_force_limit_n


@dataclass(frozen=True)
class _Verdict:
    """What a rule decided, before it is dressed up as a :class:`Decision`."""

    action: Action
    reason: str
    force: float | None = None  # None means "use the per-item grip limit"


@dataclass(frozen=True)
class _Context:
    track: Track
    tier: HazardTier
    cfg: PolicyConfig
    reachable: bool
    thermal_only_alert_k: float

    @property
    def cls_name(self) -> str:
        return CLASS_NAMES[self.track.cls_id]


def _rule_r0_hot_but_not_lithium(ctx: _Context) -> _Verdict | None:
    """Hot, but nothing says lithium. Tell a human; keep the line running."""
    if ctx.tier is not HazardTier.NONE or ctx.track.peak_dt < ctx.thermal_only_alert_k:
        return None
    return _Verdict(
        Action.ALERT_OPERATOR,
        f"{ctx.track.peak_dt:.0f} K hot object with vision score "
        f"{ctx.track.score:.2f} < {ctx.cfg.vision_review:.2f}: not lithium, no line stop",
        force=0.0,
    )


def _rule_r1_no_evidence(ctx: _Context) -> _Verdict | None:
    if ctx.tier is not HazardTier.NONE:
        return None
    return _Verdict(Action.IGNORE, "no lithium evidence", force=0.0)


def _rule_r2_runaway(ctx: _Context) -> _Verdict | None:
    """Venting or fast-rising: do not touch it, stop the belt."""
    if ctx.tier is not HazardTier.RUNAWAY:
        return None
    return _Verdict(
        Action.EMERGENCY_STOP,
        f"thermal runaway: peak {ctx.track.peak_dt:.0f} K, rise "
        f"{ctx.track.rise_rate_k_s:+.1f} K/s -> belt halt, suppression, "
        f"operator callout (no grip)",
        force=0.0,
    )


def _rule_r3_not_enough_frames(ctx: _Context) -> _Verdict | None:
    """One frame is not evidence."""
    if ctx.track.frames_seen >= ctx.cfg.min_track_frames:
        return None
    return _Verdict(
        Action.OBSERVE,
        f"seen {ctx.track.frames_seen}/{ctx.cfg.min_track_frames} frames, holding",
        force=0.0,
    )


def _rule_r4_past_the_window(ctx: _Context) -> _Verdict | None:
    """Past the pick window, or out of the workspace: flag, do not chase."""
    if ctx.track.center_m[0] <= ctx.cfg.max_reach_x_m and ctx.reachable:
        return None
    return _Verdict(
        Action.TOO_LATE,
        f"x={ctx.track.center_m[0]:.2f} m past the {ctx.cfg.max_reach_x_m:.2f} m "
        f"pick window -> flag downstream diverter",
        force=0.0,
    )


def _rule_r5_thermal_event(ctx: _Context) -> _Verdict | None:
    """Hot but stable: gentle pick straight into the quench bin."""
    if ctx.tier is not HazardTier.THERMAL_EVENT:
        return None
    return _Verdict(
        Action.EXTRACT_QUENCH,
        f"{ctx.cls_name} at {ctx.track.peak_dt:.0f} K above belt -> gentle pick to sand bin",
    )


def _rule_r6_confirmed(ctx: _Context) -> _Verdict | None:
    """Confirmed lithium item, normal extraction."""
    if ctx.tier is not HazardTier.CONFIRMED:
        return None
    detail = (
        f"vision {ctx.track.score:.2f} corroborated by {ctx.track.peak_dt:.1f} K self-heating"
        if ctx.track.score < ctx.cfg.vision_accept
        else f"vision {ctx.track.score:.2f}"
    )
    return _Verdict(Action.EXTRACT_QUARANTINE, f"{ctx.cls_name} confirmed ({detail})")


def _rule_r7_weak_hit(ctx: _Context) -> _Verdict | None:
    """Weak hit with no thermal support: keep watching, do not act."""
    return _Verdict(
        Action.OBSERVE,
        f"weak hit {ctx.track.score:.2f} with only {ctx.track.peak_dt:.1f} K rise, waiting",
        force=0.0,
    )


#: The rule set, in priority order. First match wins, and the id that matched
#: is recorded on the decision, so an auditor can read the table and the log
#: side by side. ``docs/SAFETY.md`` documents the same table in prose.
RULES: tuple[tuple[str, object], ...] = (
    ("R0", _rule_r0_hot_but_not_lithium),
    ("R1", _rule_r1_no_evidence),
    ("R2", _rule_r2_runaway),
    ("R3", _rule_r3_not_enough_frames),
    ("R4", _rule_r4_past_the_window),
    ("R5", _rule_r5_thermal_event),
    ("R6", _rule_r6_confirmed),
    ("R7", _rule_r7_weak_hit),
)


def decide(
    track: Track,
    cfg: PolicyConfig,
    *,
    reachable: bool = True,
    thermal_only_alert_k: float = 25.0,
) -> Decision:
    """Map one tracked object onto one action, with the rule that fired."""
    tier = classify(track, cfg)
    ctx = _Context(track, tier, cfg, reachable, thermal_only_alert_k)
    for rule_id, rule in RULES:
        verdict = rule(ctx)  # type: ignore[operator]
        if verdict is None:
            continue
        return Decision(
            action=verdict.action,
            tier=tier,
            rule=rule_id,
            reason=verdict.reason,
            grip_force_n=(
                grip_force_for(track, tier, cfg) if verdict.force is None else verdict.force
            ),
            priority=_priority(track, tier, cfg),
            track_id=track.track_id,
            cls_name=ctx.cls_name,
            vision_score=track.score,
            peak_dt=track.peak_dt,
            rise_rate_k_s=track.rise_rate_k_s,
        )
    raise AssertionError("the rule set must be exhaustive; R7 has no guard")


def decide_all(tracks: list[Track], cfg: PolicyConfig, *, reachable: dict[int, bool] | None = None) -> list[Decision]:
    """Decide for every track and return them most urgent first."""
    reachable = reachable or {}
    out = [decide(t, cfg, reachable=reachable.get(t.track_id, True)) for t in tracks]
    out.sort(key=lambda d: -d.priority)
    return out
