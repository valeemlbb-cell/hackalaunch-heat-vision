"""Materialise the dataset manifest and (optionally) the render cache.

The dataset is procedural: ``data/manifest.json`` records the seed ranges that
define each split, and every frame is reproducible from its seed. This script
writes the manifest, prints the split digests, and can pre-render the cache so
training does not pay the render cost.

    python scripts/make_dataset.py --train 3000 --val 400 --test 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heatvision.config import DEFAULT  # noqa: E402
from heatvision.sim.dataset import build_split, split_digest, write_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the HeatVision dataset manifest")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--digest", action="store_true", help="render each split and print its SHA-256")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest.json")
    args = ap.parse_args()

    counts = {"train": args.train, "val": args.val, "test": args.test}
    manifest = write_manifest(args.out, counts, DEFAULT)
    print(f"wrote {args.out}")
    for split, info in manifest["splits"].items():
        print(f"  {split:<5} {info['count']:>5} frames  seeds {info['seeds'][0]}..{info['seeds'][1]}")

    if args.digest:
        for split, count in counts.items():
            print(f"rendering {split} ...", flush=True)
            samples = build_split(split, count, DEFAULT, progress=True)
            n_boxes = sum(len(s.labels) for s in samples)
            print(f"  {split}: {n_boxes} labelled hazards, sha256={split_digest(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
