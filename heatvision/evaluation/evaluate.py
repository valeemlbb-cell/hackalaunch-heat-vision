"""Run the full evaluation: detection metrics, adversarial subsets, end-to-end.

Detection metrics on their own would not be an honest answer to "does this
remove batteries". The end-to-end section closes that gap by running the whole
station on unseen scenes and counting how many hazards actually made it into a
bin versus how many reached the crusher.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import DEFAULT, CellConfig
from ..detect.decode import Detection
from ..detect.infer import Detector
from ..pipeline import Station
from ..sim.dataset import Sample, build_split, split_digest
from .metrics import ImageGroundTruth, evaluate_detections, subset_report


def ground_truth_from_samples(samples: list[Sample]) -> list[ImageGroundTruth]:
    return [
        ImageGroundTruth(
            boxes=s.boxes,
            labels=s.labels,
            subsets=s.subsets,
            neg_boxes=s.neg_boxes,
            neg_subsets=s.neg_subsets,
        )
        for s in samples
    ]


@torch.no_grad()
def run_detector_on_split(
    detector: Detector, samples: list[Sample], *, batch_size: int = 32
) -> tuple[list[list[Detection]], float]:
    """Predictions in network-input pixels (same frame as ``Sample.boxes``)."""
    preds: list[list[Detection]] = []
    started = time.perf_counter()
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        images = np.stack([s.image for s in chunk]).copy()
        for ch in detector.zero_channels:
            images[:, ch] = 0.0
        batch = torch.from_numpy(images)
        preds.extend(detector.detect_tensor_batch(batch, scale=1.0))
    elapsed = time.perf_counter() - started
    return preds, elapsed


def detection_report(
    detector: Detector,
    samples: list[Sample],
    *,
    operating_threshold: float | None = None,
) -> dict:
    op = operating_threshold if operating_threshold is not None else detector.cfg.detector.score_threshold
    gts = ground_truth_from_samples(samples)
    preds, elapsed = run_detector_on_split(detector, samples)
    report = evaluate_detections(
        preds,
        gts,
        iou_thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
        operating_threshold=op,
    )
    report["subsets"] = subset_report(preds, gts, operating_threshold=op)
    report["n_images"] = len(samples)
    report["inference_ms_per_frame"] = round(elapsed / max(len(samples), 1) * 1000, 2)
    return report


def end_to_end_report(
    detector: Detector,
    cfg: CellConfig = DEFAULT,
    *,
    seeds: tuple[int, ...] = (901, 902, 903, 904, 905),
    seconds: float = 24.0,
    fps: float = 20.0,
) -> dict:
    """Run the whole cell on fresh scenes and count what actually happened."""
    runs = []
    for seed in seeds:
        station = Station(detector, cfg, seed=seed, fps=fps)
        station.run(seconds)
        runs.append(station.stats.summary())

    def agg(key: str) -> float | None:
        vals = [r[key] for r in runs if r[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    totals = {
        "hazards_presented": int(sum(r["hazards_presented"] for r in runs)),
        "hazards_removed": int(sum(r["hazards_removed"] for r in runs)),
        "hazards_reached_crusher": int(sum(r["hazards_reached_crusher"] for r in runs)),
        "pick_attempts": int(sum(r["pick_attempts"] for r in runs)),
        "false_positive_picks": int(sum(r["false_positive_picks"] for r in runs)),
        "emergency_stops": int(sum(r["emergency_stops"] for r in runs)),
        "operator_alerts": int(sum(r["operator_alerts"] for r in runs)),
        "too_late_flags": int(sum(r["too_late_flags"] for r in runs)),
    }
    totals["removal_rate"] = round(
        totals["hazards_removed"] / max(totals["hazards_presented"], 1), 4
    )
    totals["pick_success_rate"] = agg("pick_success_rate")
    totals["mean_cycle_s"] = agg("mean_cycle_s")
    totals["mean_detect_ms"] = agg("mean_detect_ms")
    return {
        "seconds_per_run": seconds,
        "fps": fps,
        "runs": len(seeds),
        "totals": totals,
        "per_run": runs,
    }


def full_report(
    checkpoint: Path,
    *,
    n_test: int = 400,
    cfg: CellConfig = DEFAULT,
    out_path: Path | None = None,
    end_to_end: bool = True,
    verbose: bool = True,
) -> dict:
    detector = Detector.load(checkpoint, cfg)
    if verbose:
        print(f"rendering held-out test split ({n_test} frames) ...", flush=True)
    samples = build_split("test", n_test, cfg, progress=verbose)
    digest = split_digest(samples)

    if verbose:
        print("scoring detections ...", flush=True)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_meta": detector.meta,
        "test_split": {
            "n": n_test,
            "seed_base": 9_000_000,
            "sha256": digest,
            "note": "held-out seeds, never rendered during training or validation",
        },
        "detection": detection_report(detector, samples),
    }
    if end_to_end:
        if verbose:
            print("running end-to-end station trials ...", flush=True)
        report["end_to_end"] = end_to_end_report(detector, cfg)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_markdown(report: dict) -> str:
    """Human-readable summary, pasted straight into the README."""
    det = report["detection"]
    lines: list[str] = []
    lines.append("### Detection (held-out test split)\n")
    lines.append(f"- frames: {det['n_images']}  |  operating threshold: {det['operating_threshold']}")
    lines.append(f"- **mAP@0.5: {det['mAP@0.5']:.3f}**  |  mAP@[0.5:0.95]: {det['mAP@[0.5:0.95]']:.3f}")
    lines.append(f"- inference: {det['inference_ms_per_frame']:.1f} ms/frame (CPU)\n")
    lines.append("| class | AP@0.5 | precision | recall | GT |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, row in det["by_iou"]["0.50"]["per_class"].items():
        lines.append(
            f"| {name} | {row['ap']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['n_gt']} |"
        )
    agn = det["by_iou"]["0.50"]["class_agnostic"]
    lines.append(
        f"| **any lithium** | - | **{agn['precision']:.3f}** | **{agn['recall']:.3f}** | {agn['n_gt']} |\n"
    )

    lines.append("### Adversarial subsets\n")
    lines.append("| subset | n | recall / false-alarm |")
    lines.append("| --- | --- | --- |")
    for key, row in det["subsets"].items():
        value = row.get("recall", row.get("false_alarm_rate"))
        lines.append(f"| {key} | {row['n']} | {value:.3f} |")
    lines.append("")

    if "end_to_end" in report:
        e2e = report["end_to_end"]["totals"]
        lines.append("### End-to-end removal (closed loop, unseen scenes)\n")
        lines.append(f"- hazards presented: {e2e['hazards_presented']}")
        lines.append(
            f"- **removed into a bin: {e2e['hazards_removed']} "
            f"({e2e['removal_rate'] * 100:.1f}%)**"
        )
        lines.append(f"- reached the crusher: {e2e['hazards_reached_crusher']}")
        lines.append(f"- pick attempts: {e2e['pick_attempts']} (success {e2e['pick_success_rate']:.2f})")
        lines.append(f"- false-positive picks: {e2e['false_positive_picks']}")
        lines.append(
            f"- emergency stops: {e2e['emergency_stops']}  |  operator alerts: {e2e['operator_alerts']}"
        )
        lines.append(f"- mean pick cycle: {e2e['mean_cycle_s']} s\n")
    return "\n".join(lines)
