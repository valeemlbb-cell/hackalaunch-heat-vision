"""Render the short looping console clip used at the top of the README.

Same :class:`~heatvision.pipeline.Station`, same detector, same rules as the
full demo — just a few seconds of it, scaled down, so a reader sees the system
working before deciding whether to click the video.

    python scripts/make_hero_gif.py --seconds 7 --out docs/hero.gif
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heatvision.config import DEFAULT  # noqa: E402
from heatvision.detect.infer import Detector, default_checkpoint  # noqa: E402
from heatvision.pipeline import Station  # noqa: E402
from heatvision.viz.overlay import attach_cell_items, compose_frame  # noqa: E402


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg not found on PATH")
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the README hero GIF")
    ap.add_argument("--checkpoint", type=Path, default=default_checkpoint(ROOT))
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--hazard-rate", type=float, default=0.5)
    ap.add_argument("--warmup", type=float, default=6.0, help="belt seconds to skip first")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "hero.gif")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 2

    detector = Detector.load(args.checkpoint, DEFAULT)
    station = Station(
        detector, DEFAULT, seed=args.seed, fps=float(args.fps), hazard_rate=args.hazard_rate
    )

    note = (
        f"stress-test scenario: {args.hazard_rate * 100:.0f}% of items are lithium hazards, "
        f"far above a real waste stream - nothing else is staged"
    )
    for _ in range(int(args.warmup * args.fps)):
        station.step()

    frames_dir = ROOT / "demo_build" / "hero"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    events: list[str] = []
    n = int(args.seconds * args.fps)
    height = int(args.width * 1080 / 1920)
    print(f"rendering {n} frames ...", flush=True)
    for i in range(n):
        record = station.step()
        events.extend(record.events)
        attach_cell_items(record, station.scene)
        frame = compose_frame(record, station.stats.summary(), DEFAULT, events=events, note=note)
        small = cv2.resize(frame, (args.width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frames_dir / f"f{i:04d}.png"), np.ascontiguousarray(small))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    palette = frames_dir / "palette.png"
    ff = ffmpeg_bin()
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-i", str(frames_dir / "f%04d.png"),
         "-vf", "palettegen=max_colors=96:stats_mode=diff", str(palette)],
        check=True,
    )
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", str(frames_dir / "f%04d.png"), "-i", str(palette),
         "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(args.out)],
        check=True,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB, {n} frames at {args.fps} fps)")
    if size_mb > 8:
        print("WARNING: that is large for a README; lower --width or --seconds", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
