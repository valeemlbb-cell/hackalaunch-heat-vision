"""Train HeatNet-S.

    python scripts/train.py --train 2400 --val 300 --epochs 12
    python scripts/train.py --ablation rgb   # thermal channel blanked
    python scripts/train.py --ablation lwir  # colour channels blanked
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heatvision.config import DEFAULT, TrainConfig  # noqa: E402
from heatvision.detect.train import train  # noqa: E402

ABLATIONS: dict[str, tuple[int, ...]] = {
    "fused": (),
    "rgb": (3,),  # blank the LWIR channel
    "lwir": (0, 1, 2),  # blank the colour channels
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the HeatVision detector")
    ap.add_argument("--train", type=int, default=2400, help="training frames")
    ap.add_argument("--val", type=int, default=300, help="validation frames")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ablation", choices=sorted(ABLATIONS), default="fused")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    args = ap.parse_args()

    out = args.out or (ROOT / "runs" / args.ablation)
    cache = args.cache if str(args.cache) not in ("", ".") else None
    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed
    )
    report = train(
        out,
        n_train=args.train,
        n_val=args.val,
        cfg=DEFAULT,
        tcfg=tcfg,
        cache_dir=cache,
        zero_channels=ABLATIONS[args.ablation],
    )
    print(
        f"\ndone: best val loss {report['best_val_loss']:.4f} "
        f"in {report['train_seconds']:.0f}s -> {report['checkpoint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
