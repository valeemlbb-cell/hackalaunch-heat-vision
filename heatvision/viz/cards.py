"""Title and results cards for the demo video."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX
BG = (18, 17, 16)
TEXT = (236, 236, 234)
MUTED = (150, 148, 146)
ACCENT = (70, 160, 250)
GOOD = (120, 220, 90)


def _blank(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:] = BG
    # subtle thermal-gradient strip so the card does not look like a slide
    strip = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(6, axis=0)
    band = cv2.applyColorMap(strip, cv2.COLORMAP_INFERNO)
    canvas[height - 6 :, :] = band
    return canvas


def title_card(
    width: int = 1920,
    height: int = 1080,
    *,
    title: str = "HeatVision",
    subtitle: str = "bispectral lithium-hazard detection and robotic removal",
    bullets: tuple[str, ...] = (),
) -> np.ndarray:
    canvas = _blank(width, height)
    cv2.putText(canvas, title, (140, 360), FONT, 3.4, TEXT, 3, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (148, 430), FONT_S, 1.0, ACCENT, 1, cv2.LINE_AA)
    y = 540
    for line in bullets:
        cv2.circle(canvas, (156, y - 8), 4, ACCENT, -1)
        cv2.putText(canvas, line, (180, y), FONT_S, 0.82, MUTED, 1, cv2.LINE_AA)
        y += 58
    return canvas


def metrics_card(
    report: dict, width: int = 1920, height: int = 1080, *, heading: str = "Measured, not claimed"
) -> np.ndarray:
    canvas = _blank(width, height)
    cv2.putText(canvas, heading, (140, 180), FONT, 1.9, TEXT, 2, cv2.LINE_AA)
    det = report.get("detection", {})
    e2e = report.get("end_to_end", {}).get("totals", {})
    rows = [
        ("held-out test frames", str(det.get("n_images", "-"))),
        ("mAP@0.5", f"{det.get('mAP@0.5', 0):.3f}"),
        ("mAP@[0.5:0.95]", f"{det.get('mAP@[0.5:0.95]') or 0:.3f}"),
        ("inference (CPU)", f"{det.get('inference_ms_per_frame', 0):.1f} ms/frame"),
        ("", ""),
        ("hazards presented (closed loop)", str(e2e.get("hazards_presented", "-"))),
        ("removed into a bin", f"{e2e.get('hazards_removed', '-')}  ({e2e.get('removal_rate', 0) * 100:.0f}%)"),
        ("reached the crusher", str(e2e.get("hazards_reached_crusher", "-"))),
        ("false-positive picks", str(e2e.get("false_positive_picks", "-"))),
        ("emergency stops", str(e2e.get("emergency_stops", "-"))),
    ]
    y = 290
    for label, value in rows:
        if not label:
            y += 26
            continue
        cv2.putText(canvas, label, (150, y), FONT_S, 0.92, MUTED, 1, cv2.LINE_AA)
        color = GOOD if "removed" in label or "mAP" in label else TEXT
        cv2.putText(canvas, value, (1080, y), FONT, 0.98, color, 1, cv2.LINE_AA)
        y += 66
    cv2.putText(
        canvas,
        "every number on this card is produced by scripts/evaluate.py on seeds the model never trained on",
        (150, height - 120),
        FONT_S,
        0.66,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    return canvas


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a lower-third caption onto a console frame."""
    if not text:
        return frame
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT_S, 0.78, 1)
    x = (w - tw) // 2
    y = h - 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 24, y - th - 20), (x + tw + 24, y + 16), (12, 12, 12), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
    cv2.putText(frame, text, (x, y), FONT_S, 0.78, TEXT, 1, cv2.LINE_AA)
    return frame
