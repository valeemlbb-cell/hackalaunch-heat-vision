"""Operator-console rendering for the demo video.

Three synchronised views — what the colour camera sees, what the LWIR camera
sees, and where the arm physically is — plus the decision log, so a viewer can
check that the action the arm took matches the rule that fired.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..arm.kinematics import elbow_position, forward
from ..config import DEFAULT, CellConfig
from ..decide.policy import Action, Decision
from ..pipeline import FrameRecord
from ..sim.render import thermal_to_color

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

BG = (18, 17, 16)
PANEL = (30, 29, 28)
EDGE = (62, 60, 58)
TEXT = (232, 232, 230)
MUTED = (150, 148, 146)

ACTION_COLOR: dict[Action, tuple[int, int, int]] = {
    Action.IGNORE: (110, 110, 110),
    Action.OBSERVE: (60, 190, 240),
    Action.EXTRACT_QUARANTINE: (120, 220, 90),
    Action.EXTRACT_QUENCH: (40, 170, 255),
    Action.EMERGENCY_STOP: (60, 60, 250),
    Action.ALERT_OPERATOR: (230, 120, 230),
    Action.TOO_LATE: (70, 130, 220),
}


def _panel(canvas: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), EDGE, 1)
    cv2.putText(canvas, title, (x + 12, y - 10), FONT, 0.52, MUTED, 1, cv2.LINE_AA)


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.45) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT_S, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 8, y + 3), (20, 20, 20), -1)
    cv2.putText(img, text, (x + 4, y - 2), FONT_S, scale, color, 1, cv2.LINE_AA)


def annotate_view(
    base: np.ndarray, record: FrameRecord, size: int, *, show_scores: bool = True
) -> np.ndarray:
    """Upscale a sensor frame and draw the current tracks on it."""
    img = cv2.resize(base, (size, size), interpolation=cv2.INTER_NEAREST)
    scale = size / base.shape[0]
    by_track = {d.track_id: d for d in record.decisions}
    for track in record.tracks:
        decision: Decision | None = by_track.get(track.track_id)
        action = decision.action if decision else Action.OBSERVE
        color = ACTION_COLOR.get(action, MUTED)
        x0, y0, x1, y1 = (int(v * scale) for v in track.box)
        thickness = 3 if action in (Action.EXTRACT_QUARANTINE, Action.EXTRACT_QUENCH, Action.EMERGENCY_STOP) else 1
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        if show_scores:
            from ..config import CLASS_NAMES

            tag = f"{CLASS_NAMES[track.cls_id]} {track.score:.2f} {track.peak_dt:+.0f}K"
            _label(img, tag, (x0, max(14, y0 - 2)), color)
    return img


def draw_cell(record: FrameRecord, cfg: CellConfig, size: int = 512) -> np.ndarray:
    """Top-down view of the physical cell: belt, arm, bins, tool."""
    img = np.full((size, size, 3), (24, 23, 22), np.uint8)
    x_min, x_max = -0.08, 1.26
    y_min, y_max = -0.78, 0.56
    span = max(x_max - x_min, y_max - y_min)
    s = size / span

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int((x - x_min) * s), int((y - y_min) * s)

    _draw_belt(img, cfg, to_px)
    _draw_bins(img, cfg, to_px)
    for item in record_items(record):
        px, py = to_px(item[0], item[1])
        cv2.circle(img, (px, py), max(2, int(item[2] * s)), item[3], -1)
    _draw_arm(img, record, cfg, to_px)
    _draw_cell_banner(img, record, size)
    return img


def _draw_belt(img: np.ndarray, cfg: CellConfig, to_px) -> None:  # noqa: ANN001 - local projector
    belt = cfg.belt
    p0 = to_px(0.0, -0.5 * belt.width_m)
    p1 = to_px(belt.view_len_m, 0.5 * belt.width_m)
    cv2.rectangle(img, p0, p1, (44, 43, 42), -1)
    cv2.rectangle(img, p0, p1, (78, 76, 74), 1)
    for x in np.arange(0.0, belt.view_len_m, 0.06):
        cv2.line(
            img, to_px(x, -0.5 * belt.width_m), to_px(x, 0.5 * belt.width_m), (52, 51, 50), 1
        )
    cx0 = to_px(belt.crusher_x_m, -0.5 * belt.width_m)
    cx1 = to_px(belt.crusher_x_m + 0.1, 0.5 * belt.width_m)
    cv2.rectangle(img, cx0, cx1, (24, 24, 96), -1)
    cv2.putText(
        img, "CRUSHER", (cx0[0] - 66, cx0[1] - 8), FONT_S, 0.38, (90, 90, 220), 1, cv2.LINE_AA
    )


def _draw_bins(img: np.ndarray, cfg: CellConfig, to_px) -> None:  # noqa: ANN001
    for center, name, color in (
        (cfg.arm.quarantine_bin_m, "QUARANTINE", (120, 220, 90)),
        (cfg.arm.quench_bin_m, "QUENCH", (40, 170, 255)),
    ):
        a = to_px(center[0] - 0.10, center[1] - 0.09)
        b = to_px(center[0] + 0.10, center[1] + 0.09)
        cv2.rectangle(img, a, b, (38, 37, 36), -1)
        cv2.rectangle(img, a, b, color, 1)
        cv2.putText(img, name, (a[0], a[1] - 6), FONT_S, 0.36, color, 1, cv2.LINE_AA)


def _draw_arm(img: np.ndarray, record: FrameRecord, cfg: CellConfig, to_px) -> None:  # noqa: ANN001
    q1, q2 = record.arm_joints[0], record.arm_joints[1]
    bp = to_px(*cfg.arm.base_xy_m)
    ep = to_px(*elbow_position(cfg.arm, q1))
    tp = to_px(*forward(cfg.arm, q1, q2))
    cv2.line(img, bp, ep, (186, 182, 176), 7)
    cv2.line(img, ep, tp, (150, 148, 144), 5)
    cv2.circle(img, bp, 9, (210, 206, 200), -1)
    cv2.circle(img, ep, 6, (120, 118, 114), -1)
    tool_color = (120, 220, 90) if record.holding else (240, 240, 240)
    cv2.circle(img, tp, 7, tool_color, 2)
    # the filled dot grows as the tool descends, so height is readable top-down
    z_frac = record.arm_joints[3] / max(cfg.arm.z_clear_m, 1e-6)
    cv2.circle(img, tp, max(2, int(9 * (1.0 - min(z_frac, 1.0)) + 2)), tool_color, -1)


def _draw_cell_banner(img: np.ndarray, record: FrameRecord, size: int) -> None:
    banner = f"belt {record.belt_speed:.2f} m/s   arm {record.arm_state}"
    color = TEXT
    if record.estop:
        cv2.rectangle(img, (0, size - 30), (size, size), (40, 40, 200), -1)
        banner = "EMERGENCY STOP - BELT HALTED"
        color = (255, 255, 255)
    cv2.putText(img, banner, (12, size - 10), FONT_S, 0.46, color, 1, cv2.LINE_AA)


def record_items(record: FrameRecord) -> list[tuple[float, float, float, tuple[int, int, int]]]:
    """Item markers for the cell view, taken from the attached scene snapshot."""
    return getattr(record, "_cell_items", [])


def attach_cell_items(record: FrameRecord, scene) -> None:  # noqa: ANN001 - simple bridge
    """Copy item positions onto the record so the cell view can draw them."""
    markers = []
    for item in scene.visible():
        color = (90, 200, 250) if item.is_hazard else (120, 118, 114)
        if item.dt_core > 22.0:
            color = (60, 90, 250)
        markers.append((item.x_m, item.y_m, 0.5 * item.long_m, color))
    record._cell_items = markers  # type: ignore[attr-defined]


PANEL_Y = 94
PANEL_SIZE = 512
LOG_Y = 630


def _draw_header(canvas: np.ndarray, record: FrameRecord, width: int, note: str = "") -> None:
    cv2.putText(canvas, "HeatVision", (24, 44), FONT, 1.0, TEXT, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "bispectral lithium-hazard detection and robotic removal  |  simulated sort line",
        (232, 44),
        FONT_S,
        0.52,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"t = {record.time_s:6.2f} s", (width - 220, 44), FONT_S, 0.6, TEXT, 1, cv2.LINE_AA
    )
    if note:
        # Kept on screen for the whole run: a viewer must never have to take
        # the README's word for how the scenario was weighted.
        cv2.putText(canvas, note, (24, 66), FONT_S, 0.46, (70, 160, 250), 1, cv2.LINE_AA)


def _draw_views(canvas: np.ndarray, record: FrameRecord, cfg: CellConfig) -> None:
    titles = (
        "RGB CAMERA",
        "LWIR THERMAL   black = belt baseline, white = +18 K",
        "CELL - TOP DOWN",
    )
    views = (
        annotate_view(record.rgb, record, PANEL_SIZE),
        annotate_view(
            thermal_to_color(record.thermal, 18.0, baseline=record.baseline_dt),
            record,
            PANEL_SIZE,
        ),
        draw_cell(record, cfg, PANEL_SIZE),
    )
    for x, title, view in zip((24, 560, 1096), titles, views):
        _panel(canvas, x, PANEL_Y, PANEL_SIZE, PANEL_SIZE, title)
        canvas[PANEL_Y : PANEL_Y + PANEL_SIZE, x : x + PANEL_SIZE] = view


def _draw_counters(canvas: np.ndarray, record: FrameRecord, stats: dict, width: int) -> None:
    col_x = 1632
    _panel(canvas, col_x, PANEL_Y, width - col_x - 24, PANEL_SIZE, "LIVE")
    rows = (
        ("hazards seen", stats.get("hazards_presented", 0)),
        ("removed", stats.get("hazards_removed", 0)),
        ("to crusher", stats.get("hazards_reached_crusher", 0)),
        ("pick attempts", stats.get("pick_attempts", 0)),
        ("false picks", stats.get("false_positive_picks", 0)),
        ("e-stops", stats.get("emergency_stops", 0)),
        ("alerts", stats.get("operator_alerts", 0)),
        ("detect ms", stats.get("mean_detect_ms") or 0),
        ("belt baseline", f"{record.baseline_dt:+.1f} K"),
    )
    yy = PANEL_Y + 46
    for name, value in rows:
        cv2.putText(canvas, name, (col_x + 16, yy), FONT_S, 0.46, MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(value), (col_x + 180, yy), FONT, 0.52, TEXT, 1, cv2.LINE_AA)
        yy += 42
    rate = stats.get("removal_rate", 0.0) or 0.0
    cv2.putText(canvas, "REMOVAL RATE", (col_x + 16, yy + 24), FONT_S, 0.46, MUTED, 1, cv2.LINE_AA)
    cv2.putText(
        canvas, f"{rate * 100:.0f}%", (col_x + 16, yy + 78), FONT, 1.5, (120, 220, 90), 2, cv2.LINE_AA
    )


def _elide(text: str, limit: int) -> str:
    """Trim to ``limit`` columns on a word boundary, so nothing ends mid-word."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 3]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,:;-") + "..."


def _draw_tables(
    canvas: np.ndarray, record: FrameRecord, events: list[str], width: int, height: int
) -> None:
    log_h = height - LOG_Y - 24
    _panel(canvas, 24, LOG_Y, 1080, log_h, "DECISIONS THIS CYCLE  (rule -> action)")
    header = (
        f"{'trk':>4}  {'class':<7} {'vis':>5} {'dT':>7} {'rate':>7}  "
        f"{'rule':<5} {'action':<20} reason"
    )
    yy = LOG_Y + 34
    cv2.putText(canvas, header, (40, yy), FONT_S, 0.42, MUTED, 1, cv2.LINE_AA)
    yy += 26
    for decision in record.decisions[:9]:
        line = (
            f"{decision.track_id:>4}  {decision.cls_name:<7} {decision.vision_score:>5.2f} "
            f"{decision.peak_dt:>+6.1f}K {decision.rise_rate_k_s:>+6.1f}  "
            f"{decision.rule:<5} {decision.action.value:<20} {_elide(decision.reason, 58)}"
        )
        cv2.putText(
            canvas, line, (40, yy), FONT_S, 0.42, ACTION_COLOR.get(decision.action, TEXT), 1, cv2.LINE_AA
        )
        yy += 24

    _panel(canvas, 1128, LOG_Y, width - 1128 - 24, log_h, "EVENT LOG")
    yy = LOG_Y + 34
    for line in events[-11:]:
        cv2.putText(canvas, _elide(line, 62), (1144, yy), FONT_S, 0.42, TEXT, 1, cv2.LINE_AA)
        yy += 24


def compose_frame(
    record: FrameRecord,
    stats: dict,
    cfg: CellConfig = DEFAULT,
    *,
    width: int = 1920,
    height: int = 1080,
    events: list[str] | None = None,
    note: str = "",
) -> np.ndarray:
    """The full 1080p operator console for one control cycle."""
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = BG
    _draw_header(canvas, record, width, note)
    _draw_views(canvas, record, cfg)
    _draw_counters(canvas, record, stats, width)
    _draw_tables(canvas, record, events or [], width, height)
    return canvas
