# REVIEW_2 — deliverables checklist audit

**Packet:** heat-vision · **Reviewed:** 2026-09-24 ~03:45 WIB · **Reviewer:** Judge 2 (deliverables auditor)
**Rules re-read:** https://hackalaunch.com/h/heat-vision — close Sep 28 13:15 UTC; 24h token-holder vote; pool ~11.6 SOL + all future fees.

## Checklist against the five required deliverables

| # | Requirement | Verdict |
| --- | --- | --- |
| 1 | Public GitHub repo, README covering execution + hardware/simulator + training method | **PASS on content, NOT PUSHED.** README is excellent on all three. `gh repo create warung-ops/heatvision --public --source=. --push` is written in RUN.md as required. 11 local commits. |
| 2 | Model results: dataset sources, held-out test set, detection metrics (P/R/mAP) + "how many hazards it missed" | **FAIL.** `results/` does not exist. README `METRICS:BEGIN/END`, `ABLATION:BEGIN/END` and `TESTS:BEGIN/END` are still placeholder stubs saying "Run `python scripts/evaluate.py`". Training is mid-flight (`runs/train_fused.log` at epoch 11/14); no rgb/lwir ablation runs exist. Numbers are the single biggest thing a judge checks and there are currently none. |
| 3 | Demo video ≤3 min showing full loop on a cluttered scene: detect → decide → physically remove | **FAIL.** No `.mp4` anywhere. Only `demo_build/_smoke_console.png` and 7 VO wavs. `<DEMO_VIDEO_URL>` unreplaced in README. Rules disqualify "non-functional demos"; a missing video is an automatic loss. |
| 4 | Brief project description | **PASS.** README hero + problem framing is submission-form ready; no separate `DESCRIPTION.md` for paste-in though. |
| 5 | No API keys, `.env.example`, large weights linked externally | **PARTIAL.** `.env.example` present and clean; grep for key/secret/token patterns finds only prose. But `.gitignore` excludes `*.pt` and `runs/`, so `heatnet_s.pt` (5.2 MB) ships nowhere and **no external weights link exists** — a fresh clone cannot reproduce the metrics without a 20-min retrain. |

## Other checks

- **Tests actually run and pass:** verified live — `python -m pytest -q` = **166 passed, 0 failed**. (README prose says 151; stale count.)
- **Pre-hackathon work marked:** PASS, and strongly — `docs/AUTONOMY.md` plus a README "Pre-hackathon work: none" section that names the disqualification clause it answers.
- **Safety / no real cells:** PASS, stated explicitly twice.
- **Licence:** PASS — MIT, no bundled assets, Hershey fonts only, 100% self-generated dataset so no dataset-licence exposure.
- **Autonomy honesty:** PASS and unusually good — grasp success is *verified* not assumed, `grasp_miss` and `too_late_flags` are surfaced rather than hidden.
- **Repo hygiene:** `__pycache__/` and `.pytest_cache/` exist on disk but are git-ignored; fine.
- **`results/` is git-ignored** while RUN.md's own submission checklist demands "`results/METRICS.md` present" — those two contradict. Metrics files must be force-added or moved out of the ignore.

## Verdict

The engineering is submission-winning; the *packet* is not yet submittable. Three of five deliverables are blocked on one thing finishing: training → evaluate → record. Nothing here needs new code.

## Concrete fixes, in order

1. Let `scripts/train.py` finish (epoch 11/14), then run the two ablations — without `rgb`/`lwir` runs the ablation table cannot be filled and the fusion claim is unevidenced.
2. `python scripts/evaluate.py --checkpoint runs/fused/heatnet_s.pt --test 400` and `--ablations`, then `python scripts/update_readme.py` to fill all three marker blocks.
3. Make sure the metrics output states **absolute missed-hazard counts** ("N of M hazards reached the crusher"), not only mAP — the rules ask for that phrasing literally.
4. Un-ignore the metrics: drop `results/` from `.gitignore` (or `git add -f results/METRICS.md results/metrics.json`). Keep `runs/` ignored.
5. `python scripts/record_demo.py --seconds 88 --out demo.mp4`; confirm duration < 3:00 and that a cluttered frame, the fired rule id, and the arm actually lifting the item are all visible on screen.
6. Upload the video + attach `heatnet_s.pt` to a GitHub release; replace `<DEMO_VIDEO_URL>` and add a "Pretrained weights" link line to the README Quick start.
7. Fix the stale "151 tests" prose to 166 (or let `update_readme.py` write it).
8. Add `DESCRIPTION.md` — 120 words, ready to paste into the submission form, so the description is not improvised at 13:00 UTC.
9. Run the RUN.md history grep for secrets once before the push, and verify a fresh `git clone` + `pip install -r requirements.txt` + `pytest` is green.
