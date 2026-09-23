"""Record the demo video: live console capture + narration.

Nothing is staged. The recorder runs the same :class:`~heatvision.pipeline.Station`
the evaluation runs, composes the operator console for every control cycle, and
pipes the frames straight into ffmpeg. The narration is a fixed timeline that
describes the *system*; it never claims a specific event happened, because the
events are emergent and differ run to run.

    python scripts/record_demo.py --seconds 100 --out demo.mp4
    python scripts/record_demo.py --no-audio      # skip TTS
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heatvision.config import DEFAULT  # noqa: E402
from heatvision.detect.infer import CHECKPOINT_NAME, Detector  # noqa: E402
from heatvision.pipeline import Station  # noqa: E402
from heatvision.viz.cards import caption, metrics_card, title_card  # noqa: E402
from heatvision.viz.overlay import attach_cell_items, compose_frame  # noqa: E402

VIDEO_FPS = 25
VOICE = "en-US-AndrewNeural"
FALLBACK_VOICE = "en-US-GuyNeural"

#: (start_second, caption, narration). Start times are wall-clock in the
#: finished video, so the audio lines up with the phase on screen.
NARRATION: tuple[tuple[float, str, str], ...] = (
    (
        0.0,
        "",
        "Heat Vision. A sorting station that finds lithium batteries in a waste stream "
        "and takes them off the belt before the crusher.",
    ),
    (
        8.0,
        "Three views: colour camera, thermal camera, and the physical cell",
        "Three views. On the left, the colour camera looking down at the belt. In the middle, "
        "a long wave infrared camera, showing temperature above the belt surface. On the right, "
        "the cell itself: the belt, the arm, the quarantine bin, the quench bin, and the crusher inlet.",
    ),
    (
        22.0,
        "One network, four channels: red, green, blue and calibrated thermal",
        "Detection is a single small network with four input channels. Red, green, blue, and thermal, "
        "fused at the first layer. One point three million parameters, running on a laptop C P U, "
        "about thirty milliseconds a frame.",
    ),
    (
        36.0,
        "Decision logic: vision says what it is, thermal says how dangerous it is",
        "The boxes are not the decision. Every track goes through an explicit policy. Vision decides "
        "what the object is. Thermal decides how dangerous it is right now. A weak visual hit that is "
        "also self heating gets promoted. A very hot object with no lithium evidence does not stop the "
        "line, it raises an operator alert, because that is a brake disc, not a cell.",
    ),
    (
        56.0,
        "The arm intercepts a moving target, then verifies the grasp",
        "When the policy calls for extraction, the arm solves an intercept for a target that is still "
        "moving, flies a trajectory inside its acceleration limits, and closes the gripper. Then it "
        "checks whether it actually got it. A miss is recorded as a miss. Nothing here is scripted.",
    ),
    (
        74.0,
        "Venting cell: stop the belt, do not grip it",
        "If a cell is venting, above the runaway threshold or rising faster than three and a half kelvin "
        "a second, the arm does not touch it. The belt stops, suppression is called, and a human is "
        "brought in. Squeezing a cell that is already failing is how you start a fire.",
    ),
    (
        92.0,
        "Every number is measured on seeds the model never trained on",
        "Finally, the numbers. Detection metrics come from a held out split of seeds the model has never "
        "seen. The removal rate comes from running this same closed loop on fresh scenes and counting how "
        "many hazards reached a bin, and how many reached the crusher. Repository, code and evaluation "
        "scripts are linked in the description.",
    ),
)


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
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def caption_at(t: float) -> str:
    text = ""
    for start, cap, _vo in NARRATION:
        if t >= start:
            text = cap
    return text


async def _synth(text: str, path: Path, voice: str) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(path))


def synthesise(out_dir: Path) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[float, Path]] = []
    for i, (start, _cap, text) in enumerate(NARRATION):
        path = out_dir / f"vo_{i:02d}.mp3"
        for voice in (VOICE, FALLBACK_VOICE):
            try:
                asyncio.run(_synth(text, path, voice))
                break
            except Exception as exc:  # noqa: BLE001 - network TTS is best effort
                print(f"  voice {voice} failed: {exc}")
        if path.exists() and path.stat().st_size > 0:
            clips.append((start, path))
    return clips


def mux(video: Path, clips: list[tuple[float, Path]], out: Path) -> None:
    if not clips:
        shutil.copyfile(video, out)
        return
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(video)]
    for _start, path in clips:
        cmd += ["-i", str(path)]
    parts = []
    for i, (start, _path) in enumerate(clips, start=1):
        parts.append(f"[{i}:a]adelay={int(start * 1000)}|{int(start * 1000)}[a{i}]")
    mixed = "".join(f"[a{i}]" for i in range(1, len(clips) + 1))
    parts.append(f"{mixed}amix=inputs={len(clips)}:normalize=0:dropout_transition=0[aout]")
    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Record the HeatVision demo video")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "fused" / CHECKPOINT_NAME)
    ap.add_argument("--seconds", type=float, default=88.0, help="simulated seconds to record")
    ap.add_argument("--fps", type=float, default=20.0, help="control-loop rate")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--hazard-rate", type=float, default=0.5, help="stress-test hazard share")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "demo.mp4")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 2

    detector = Detector.load(args.checkpoint, DEFAULT)
    station = Station(
        detector, DEFAULT, seed=args.seed, fps=args.fps, hazard_rate=args.hazard_rate
    )

    width, height = 1920, 1080
    work = ROOT / "demo_build"
    work.mkdir(parents=True, exist_ok=True)
    silent = work / "console_silent.mp4"
    writer = open_writer(silent, width, height, VIDEO_FPS)
    assert writer.stdin is not None

    def push(frame: np.ndarray, count: int = 1) -> None:
        data = np.ascontiguousarray(frame).tobytes()
        for _ in range(count):
            writer.stdin.write(data)

    # ---- intro card (6 s)
    push(
        title_card(
            width,
            height,
            bullets=(
                "detect lithium cells, pouches and devices in cluttered waste",
                "fuse a colour camera with a long-wave infrared camera",
                "decide with an explicit, auditable safety policy",
                "physically remove with a 4-DOF arm, and verify the grasp",
            ),
        ),
        6 * VIDEO_FPS,
    )

    n_steps = int(round(args.seconds * args.fps))
    repeat = max(1, int(round(VIDEO_FPS / args.fps)))
    event_log: list[str] = []
    print(f"recording {n_steps} control cycles ({args.seconds:.0f} s of belt time) ...", flush=True)
    for i in range(n_steps):
        record = station.step()
        event_log.extend(record.events)
        attach_cell_items(record, station.scene)
        video_t = 6.0 + i / args.fps
        frame = compose_frame(
            record, station.stats.summary(), DEFAULT, width=width, height=height, events=event_log
        )
        frame = caption(frame, caption_at(video_t))
        push(frame, repeat)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_steps}", flush=True)

    # ---- results card (10 s)
    report: dict = {}
    if args.metrics.exists():
        report = json.loads(args.metrics.read_text(encoding="utf-8"))
    else:
        report = {
            "detection": {},
            "end_to_end": {"totals": station.stats.summary()},
        }
    push(metrics_card(report), 10 * VIDEO_FPS)

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
        print("synthesising narration ...", flush=True)
        clips = synthesise(work / "vo")
        mux(silent, clips, args.out)

    duration = subprocess.run(
        [
            ffmpeg_bin().replace("ffmpeg", "ffprobe"),
            "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(args.out),
        ],
        capture_output=True, text=True,
    ).stdout.strip()
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({duration}s, {size_mb:.1f} MB)")
    if duration and float(duration) > 180:
        print("WARNING: video exceeds the 3 minute limit", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
