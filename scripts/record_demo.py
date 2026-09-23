"""Record the demo video: live console capture + narration.

Nothing is staged. The recorder runs the same :class:`~heatvision.pipeline.Station`
that ``scripts/evaluate.py`` runs, composes the operator console for every
control cycle, and pipes the frames straight into ffmpeg. The only additions
are a title card, a results card and a fixed narration timeline — see
``docs/AUTONOMY.md`` for the full disclosure.

    python scripts/record_demo.py --seconds 88 --out demo.mp4
    python scripts/record_demo.py --no-audio
"""

from __future__ import annotations

import argparse
import json
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
from heatvision.viz.cards import caption, metrics_card, title_card  # noqa: E402
from heatvision.viz.narration import caption_at, check_overlaps, synthesise  # noqa: E402
from heatvision.viz.overlay import attach_cell_items, compose_frame  # noqa: E402

VIDEO_FPS = 25
INTRO_S = 6
OUTRO_S = 14


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg not found on PATH")
    return exe


def open_writer(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    cmd = [
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def mux(video: Path, clips: list[tuple[float, Path]], out: Path) -> None:
    """Lay every narration clip onto the silent video at its start time."""
    if not clips:
        shutil.copyfile(video, out)
        return
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(video)]
    for _start, path in clips:
        cmd += ["-i", str(path)]
    parts = []
    for i, (start, _path) in enumerate(clips, start=1):
        ms = int(start * 1000)
        parts.append(f"[{i}:a]aresample=44100,adelay={ms}|{ms}[a{i}]")
    mixed = "".join(f"[a{i}]" for i in range(1, len(clips) + 1))
    # The narration ends well before the results card does, so the mix is
    # shorter than the video. Both obvious fixes are traps: bare -shortest cuts
    # the video off at the end of the audio and eats the results card, and a
    # bare apad never ends, so ffmpeg writes silence until the disk fills.
    # Pad to exactly the video duration and cap the output with -t.
    parts.append(f"{mixed}amix=inputs={len(clips)}:normalize=0:dropout_transition=0[amixed]")
    video_s = probe_duration(video)
    if video_s is None:
        raise SystemExit(f"cannot read the duration of {video}; is ffprobe on PATH?")
    parts.append(f"[amixed]apad=whole_dur={video_s:.3f}[aout]")
    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
        "-t", f"{video_s:.3f}", str(out),
    ]
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def probe_frames(path: Path) -> int | None:
    """Count actually decodable video frames, not what the container claims."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def record_run(
    station: Station, writer_push, args: argparse.Namespace, size: tuple[int, int]
) -> list[str]:
    """Step the station, composing and pushing one console frame per cycle."""
    width, height = size
    repeat = max(1, int(round(VIDEO_FPS / args.fps)))
    n_steps = int(round(args.seconds * args.fps))
    events: list[str] = []
    print(f"recording {n_steps} control cycles ({args.seconds:.0f} s of belt time) ...", flush=True)
    for i in range(n_steps):
        record = station.step()
        events.extend(record.events)
        attach_cell_items(record, station.scene)
        frame = compose_frame(
            record,
            station.stats.summary(),
            DEFAULT,
            width=width,
            height=height,
            events=events,
            note=(
                f"stress-test scenario: {args.hazard_rate * 100:.0f}% of items are lithium "
                f"hazards, far above a real waste stream - nothing else is staged "
                f"(docs/AUTONOMY.md)"
            ),
        )
        writer_push(caption(frame, caption_at(INTRO_S + i / args.fps)), repeat)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n_steps}", flush=True)
    return events


def load_results(path: Path, station: Station) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("no results/metrics.json yet; the results card will show this run only")
    return {"detection": {}, "end_to_end": {"totals": station.stats.summary()}}


def add_narration(silent: Path, work: Path, out: Path) -> None:
    print("synthesising narration ...", flush=True)
    clips = synthesise(work / "vo")
    for warning in check_overlaps(clips):
        print(f"  WARNING: {warning}")
    mux(silent, clips, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Record the HeatVision demo video")
    ap.add_argument("--checkpoint", type=Path, default=default_checkpoint(ROOT))
    ap.add_argument("--seconds", type=float, default=88.0, help="belt seconds to record")
    ap.add_argument(
        "--fps",
        type=float,
        default=float(VIDEO_FPS),
        help="control-loop rate; keep it equal to the video fps so belt time is wall time",
    )
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument(
        "--hazard-rate",
        type=float,
        default=0.5,
        help="stress-test hazard share; far higher than a real stream, see docs/AUTONOMY.md",
    )
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "demo.mp4")
    ap.add_argument(
        "--work",
        type=Path,
        default=ROOT / "demo_build",
        help="scratch directory for the silent cut and the narration clips; give "
        "two concurrent recordings two different directories, or they overwrite "
        "each other's frames and both videos come out corrupt",
    )
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint}; run scripts/train.py first", file=sys.stderr)
        return 2

    detector = Detector.load(args.checkpoint, DEFAULT)
    station = Station(detector, DEFAULT, seed=args.seed, fps=args.fps, hazard_rate=args.hazard_rate)

    width, height = 1920, 1080
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    silent = work / "console_silent.mp4"
    writer = open_writer(silent, width, height, VIDEO_FPS)
    assert writer.stdin is not None

    def push(frame: np.ndarray, count: int = 1) -> None:
        data = np.ascontiguousarray(frame).tobytes()
        for _ in range(count):
            writer.stdin.write(data)  # type: ignore[union-attr]

    push(
        title_card(
            width,
            height,
            bullets=(
                "detect lithium cells, pouches and devices in cluttered waste",
                "fuse a colour camera with a long-wave infrared camera",
                "decide with an explicit, auditable safety policy",
                "physically remove with a 4-DOF arm, and verify the grasp",
                "SIMULATED CELL - no physical hardware, no real lithium cells",
                f"this run is stress-weighted: {args.hazard_rate * 100:.0f}% hazards, "
                f"far above a real stream",
            ),
        ),
        INTRO_S * VIDEO_FPS,
    )

    event_log = record_run(station, push, args, (width, height))
    push(metrics_card(load_results(args.metrics, station)), OUTRO_S * VIDEO_FPS)

    writer.stdin.close()
    writer.wait()

    summary = station.stats.summary()
    (work / "run_stats.json").write_text(
        json.dumps({"stats": summary, "events": event_log}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if args.no_audio:
        shutil.copyfile(silent, args.out)
    else:
        add_narration(silent, work, args.out)

    duration = probe_duration(args.out)
    size_mb = args.out.stat().st_size / 1e6
    shown = f"{duration:.1f}" if duration else "?"
    print(f"\nwrote {args.out}  ({shown}s, {size_mb:.1f} MB)")

    # Integrity gate. A container can report a full duration while most of its
    # frames are unreadable -- that is what happens when two recordings share a
    # work directory and interleave writes into the same silent cut. Count the
    # frames that actually decode and refuse to call such a file finished.
    expected = int(round((INTRO_S + args.seconds + OUTRO_S) * VIDEO_FPS))
    frames = probe_frames(args.out)
    if frames is not None:
        print(f"decodable frames: {frames}/{expected}")
        if frames < expected * 0.98:
            print(
                f"ERROR: only {frames} of {expected} frames decode -- the video is "
                f"damaged, do NOT submit it. The usual cause is a second recorder "
                f"writing the same --work directory; re-run alone, or pass a "
                f"separate --work path.",
                file=sys.stderr,
            )
            return 1

    if duration and duration > 180:
        print("WARNING: video exceeds the 3 minute limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
