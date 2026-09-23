"""Demo narration: the script, and two ways to speak it.

The voice-over is a *fixed timeline* written before the run. It describes the
system and never claims that a particular event is happening at that moment,
because the events in the demo are emergent and differ between runs. See
``docs/AUTONOMY.md``.

Two backends, tried in order:

1. **Local SAPI** (Windows ``System.Speech``) — offline, no account, no key.
2. **edge-tts** — better voice, needs network. Used only if the local one is
   unavailable.

If neither works the recorder still produces the video, silently, with the
on-screen captions intact.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Line:
    """One narration cue: when it starts, its caption, and what is said."""

    start_s: float
    caption: str
    text: str


SCRIPT: tuple[Line, ...] = (
    Line(
        0.0,
        "",
        "Heat Vision. A sorting station that finds lithium batteries in a waste stream "
        "and takes them off the belt before the crusher.",
    ),
    Line(
        8.0,
        "Three views: colour camera, thermal camera, and the physical cell",
        "Three views. On the left, the colour camera looking down at the belt. In the middle, "
        "a long wave infrared camera, showing temperature above the belt surface. On the right, "
        "the cell itself: the belt, the arm, the two bins, and the crusher inlet.",
    ),
    Line(
        24.0,
        "One network, four channels: red, green, blue and calibrated thermal",
        "Detection is one small network with four input channels. Red, green, blue, and thermal, "
        "fused at the first layer, one point three million parameters, running on a laptop "
        "processor.",
    ),
    Line(
        36.0,
        "Vision says what it is. Thermal says how dangerous it is right now.",
        "The boxes are not the decision. Every track goes through an explicit policy. Vision decides "
        "what the object is. Thermal decides how dangerous it is right now. A weak visual hit that is "
        "also self heating gets promoted. A very hot object with no evidence of lithium does not stop "
        "the line. It raises an operator alert, because that is a brake disc, not a cell.",
    ),
    Line(
        58.0,
        "The arm intercepts a moving target, then verifies the grasp",
        "When the policy calls for extraction, the arm solves an intercept for a target that is still "
        "moving, flies a trajectory inside its acceleration limits, and closes the gripper. Then it "
        "checks whether it actually got it. A miss is recorded as a miss.",
    ),
    Line(
        73.0,
        "Venting cell: stop the belt, do not grip it",
        "If a cell is venting, the arm does not touch it. The belt stops and a human is called, because "
        "squeezing a cell that is already failing is how you start a fire.",
    ),
    Line(
        86.0,
        "Every number is measured on seeds the model never trained on",
        "Finally, the numbers. Detection metrics come from a held out split of seeds the model has never "
        "seen, and the removal rate comes from running this same loop on fresh scenes. Code, tests and "
        "the evaluation script are in the repository.",
    ),
)


def caption_at(t: float) -> str:
    """Caption that should be on screen at video time ``t``."""
    text = ""
    for line in SCRIPT:
        if t >= line.start_s:
            text = line.caption
    return text


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
_PS_TEMPLATE = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{ $s.SelectVoice('{voice}') }} catch {{ }}
$s.Rate = {rate}
$s.SetOutputToWaveFile('{out}')
$s.Speak([IO.File]::ReadAllText('{src}'))
$s.Dispose()
"""


def synth_sapi(
    text: str, out_path: Path, *, voice: str = "Microsoft David Desktop", rate: int = 2
) -> bool:
    """Speak ``text`` to a wav with the local Windows voice. Offline."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "line.txt"
        src.write_text(text, encoding="utf-8")
        script = Path(tmp) / "say.ps1"
        script.write_text(
            _PS_TEMPLATE.format(
                voice=voice,
                rate=rate,
                out=str(out_path).replace("'", "''"),
                src=str(src).replace("'", "''"),
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            capture_output=True,
            text=True,
        )
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024


def synth_edge(text: str, out_path: Path, *, voice: str = "en-US-AndrewNeural") -> bool:
    """Speak ``text`` with edge-tts. Needs network; best effort."""
    try:
        import asyncio

        import edge_tts
    except ImportError:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(edge_tts.Communicate(text, voice).save(str(out_path)))
    except Exception:  # noqa: BLE001 - offline machines are expected
        return False
    return out_path.exists() and out_path.stat().st_size > 1024


def synthesise(out_dir: Path, script: tuple[Line, ...] = SCRIPT, *, verbose: bool = True) -> list[tuple[float, Path]]:
    """Render every line, returning ``(start_second, audio_path)`` pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[float, Path]] = []
    for i, line in enumerate(script):
        wav = out_dir / f"vo_{i:02d}.wav"
        mp3 = out_dir / f"vo_{i:02d}.mp3"
        if synth_sapi(line.text, wav):
            clips.append((line.start_s, wav))
        elif synth_edge(line.text, mp3):
            clips.append((line.start_s, mp3))
        elif verbose:
            print(f"  narration line {i} could not be synthesised, skipping")
    return clips


def audio_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def check_overlaps(clips: list[tuple[float, Path]]) -> list[str]:
    """Warn if a line runs into the next one's slot."""
    warnings: list[str] = []
    for (start, path), (next_start, _next) in zip(clips, clips[1:]):
        duration = audio_duration(path)
        if duration is None:
            continue
        if start + duration > next_start + 0.25:
            warnings.append(
                f"line at {start:.0f}s runs {start + duration:.1f}s, overlapping the "
                f"{next_start:.0f}s line by {start + duration - next_start:.1f}s"
            )
    return warnings
