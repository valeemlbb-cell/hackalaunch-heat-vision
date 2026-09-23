"""Paste measured results into the README between its marker comments.

Keeping this a script rather than hand-editing means the README cannot drift
away from what ``scripts/evaluate.py`` actually produced.

    python scripts/update_readme.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heatvision.evaluation.evaluate import format_markdown  # noqa: E402

ABLATION_LABEL = {
    "fused": "**RGB + LWIR (shipped)**",
    "rgb": "RGB only",
    "lwir": "LWIR only",
}


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- {name}:BEGIN -->)(.*?)(<!-- {name}:END -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"marker {name} not found in README")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def ablation_table(data: dict) -> str:
    if not data:
        return "_Not run yet._"
    rows = [
        "| variant | mAP@0.5 | precision | recall | cold-battery recall | "
        "hot-decoy false alarm | lookalike false alarm |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def fmt(value) -> str:
        return "-" if value is None else f"{value:.3f}"

    for key in ("fused", "rgb", "lwir"):
        row = data.get(key)
        if not row:
            continue
        rows.append(
            f"| {ABLATION_LABEL[key]} | {fmt(row['mAP@0.5'])} | {fmt(row['precision'])} | "
            f"{fmt(row['recall'])} | {fmt(row['cold_battery_recall'])} | "
            f"{fmt(row['hot_decoy_false_alarm'])} | {fmt(row['lookalike_false_alarm'])} |"
        )
    return "\n".join(rows)


#: pytest.ini already carries ``-q``; a second one suppresses the count line.
_SUMMARY = re.compile(r"^[=\s]*\d+ (passed|failed)")


def test_summary() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    summary = next((line for line in reversed(lines) if _SUMMARY.match(line)), None)
    last = summary or (lines[-1] if lines else "no output")
    status = "all passing" if proc.returncode == 0 else "FAILURES"
    return f"```\n$ python -m pytest -q\n{last}\n```\n\n{status}."


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh the README result blocks")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    ap.add_argument("--ablations", type=Path, default=ROOT / "results" / "ablations.json")
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    text = args.readme.read_text(encoding="utf-8")

    if args.metrics.exists():
        report = json.loads(args.metrics.read_text(encoding="utf-8"))
        body = format_markdown(report)
        digest = report.get("test_split", {}).get("sha256", "")
        if digest:
            body += f"\nHeld-out test split SHA-256: `{digest[:32]}...`\n"
        text = replace_block(text, "METRICS", body)
        print("updated METRICS")

    if args.ablations.exists():
        data = json.loads(args.ablations.read_text(encoding="utf-8"))
        text = replace_block(text, "ABLATION", ablation_table(data))
        print("updated ABLATION")

    if not args.skip_tests:
        text = replace_block(text, "TESTS", test_summary())
        print("updated TESTS")

    args.readme.write_text(text, encoding="utf-8")
    print(f"wrote {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
