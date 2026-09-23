"""Score a checkpoint on the held-out test split and on the closed loop.

    python scripts/evaluate.py --checkpoint runs/fused/heatnet_s.pt --test 400
    python scripts/evaluate.py --ablations          # compare fused / rgb / lwir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heatvision.config import DEFAULT  # noqa: E402
from heatvision.detect.infer import CHECKPOINT_NAME, Detector, default_checkpoint  # noqa: E402
from heatvision.evaluation.evaluate import (  # noqa: E402
    detection_report,
    format_markdown,
    full_report,
)
from heatvision.sim.dataset import build_split  # noqa: E402


def run_ablations(n_test: int, out_dir: Path) -> dict:
    """Train-time ablations scored on the same held-out split."""
    samples = build_split("test", n_test, DEFAULT, progress=True)
    results: dict[str, dict] = {}
    for variant in ("fused", "rgb", "lwir"):
        ckpt = ROOT / "runs" / variant / CHECKPOINT_NAME
        if not ckpt.exists():
            print(f"skip {variant}: no checkpoint at {ckpt}")
            continue
        detector = Detector.load(ckpt, DEFAULT)
        report = detection_report(detector, samples)
        agn = report["by_iou"]["0.50"]["class_agnostic"]
        subsets = report["subsets"]
        results[variant] = {
            "mAP@0.5": report["mAP@0.5"],
            "mAP@[0.5:0.95]": report["mAP@[0.5:0.95]"],
            "precision": agn["precision"],
            "recall": agn["recall"],
            "cold_battery_recall": subsets.get("cold_battery", {}).get("recall"),
            "thermal_event_recall": subsets.get("thermal_event", {}).get("recall"),
            "hot_decoy_false_alarm": subsets.get("NEG:hot_decoy", {}).get("false_alarm_rate"),
            "lookalike_false_alarm": subsets.get("NEG:cell_lookalike", {}).get("false_alarm_rate"),
        }
        print(f"{variant:<6} mAP@0.5={report['mAP@0.5']:.3f}  P={agn['precision']:.3f}  R={agn['recall']:.3f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate HeatVision")
    ap.add_argument("--checkpoint", type=Path, default=default_checkpoint(ROOT))
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--no-end-to-end", action="store_true")
    ap.add_argument("--ablations", action="store_true", help="compare the three modality variants")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.ablations:
        run_ablations(args.test, args.out)
        return 0

    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint}; run scripts/train.py first", file=sys.stderr)
        return 2

    report = full_report(
        args.checkpoint,
        n_test=args.test,
        out_path=args.out / "metrics.json",
        end_to_end=not args.no_end_to_end,
    )
    markdown = format_markdown(report)
    (args.out / "METRICS.md").write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"\nwrote {args.out / 'metrics.json'} and {args.out / 'METRICS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
