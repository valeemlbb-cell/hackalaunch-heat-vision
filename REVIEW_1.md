# REVIEW_1 — hostile judge audit, heat-vision
Reviewed 2026-09-24 ~03:40 WIB against https://hackalaunch.com/h/heat-vision
Score: **62 / 100** (as the packet stands right now). Not disqualifiable, but
two *required* submission artifacts are missing, which is a scoring death by
checklist rather than by ethics.

## Disqualifier hunt — result: NONE FOUND
| Vector | Finding |
|---|---|
| Secrets in repo | Clean. `git grep` over full tree finds no key/token/private-key material. `.env` is git-ignored; `.env.example` present, documents only optional hardware/TTS vars, no real values. |
| Mainnet / real money | None. No chain code at all. Payout address appears only in `RUN.md` (operator note), not in shipped code. |
| Live lithium battery harmed | None. 100% synthetic; README + `docs/AUTONOMY.md` state explicitly no physical cells. |
| Pre-scripted demo sold as autonomous | Disclosed correctly in `docs/AUTONOMY.md`: narration timeline + cards scripted, everything else live, `--hazard-rate 0.5` inflation disclosed. This is the single biggest DQ vector on this hackathon and the packet handles it well. |
| Detection-only entry | No — arm + gripper + belt-halt actions present, grasp verified (`grasp_miss` path). |
| Dataset licence violation | None. Seed-addressed synthetic data; no external dataset. |
| Unlicensed assets | None bundled. OpenCV Hershey fonts only; narration synthesised at build time. MIT `LICENSE` present. |
| Plagiarism | No vendored third-party source; all modules original-looking and internally consistent. |
| Admin backdoor | None. No network listener, no remote control, no privileged path. Ground-truth access is confined to `heatvision/pipeline.py` and each site is marked `SIMULATION ONLY`; it is not visible to detector/tracker/policy. |
| Human-approval gate | N/A — packet has no outreach/posting/spend feature. |
| Pre-hackathon work marking | Declared "none", git history all 2026-09-24, consistent. |

Tests: `python -m pytest -q` → **166 passed**, clean.

## BLOCKERS (fix before submission)
1. **No metrics.** The rules require precision, recall and "how many hazards it
   missed". `results/` does not exist; README still shows
   `<!-- METRICS:BEGIN -->` / `<!-- ABLATION:BEGIN -->` / `<!-- TESTS:BEGIN -->`
   placeholders. Training is still mid-run (`runs/train_fused.log` at epoch
   11/14). A judge opening the repo today sees a results section that says
   "Run python scripts/evaluate.py to regenerate".
2. **No demo video.** No `demo.mp4` anywhere; README ships the literal string
   `<DEMO_VIDEO_URL>`. A ≤3 min loop video is a hard requirement.
3. **The public repo is stale and already pushed.** `origin` =
   `https://github.com/valeemlbb-cell/hackalaunch-heat-vision.git`, whose
   `master` sits at `4b72a55`, one commit behind local `8b48a5d`, and four
   tracked files are modified-but-uncommitted (`RUN.md`,
   `heatvision/detect/train.py`, `requirements.txt`, `scripts/record_demo.py`).
   Whatever a judge clones is not what was tested here.
4. **RUN.md publish line contradicts reality.** It says
   `gh repo create warung-ops/heatvision --public --source=. --push`, but the
   repo already exists under a different owner/name. Two different repos is the
   fastest way to submit a URL that 404s or is empty.

## HIGH
5. **Weights are unobtainable from a clean clone.** `*.pt` and `runs/` are
   git-ignored, and the rules say large weights must be linked externally
   (e.g. Hugging Face). README's Quick start tells a judge to run
   `evaluate.py --checkpoint runs/fused/heatnet_s.pt`, which does not exist
   after `git clone`. Either link the checkpoint externally or state plainly
   that evaluation requires a ~20 min training run first.
6. **Metric tables must be reproducible on demand.** Once evaluation runs,
   commit `results/metrics.json`, `results/METRICS.md` and
   `results/ablations.json` — they are git-ignored today via `results/` in
   `.gitignore`, so the numbers the README quotes would not ship.
7. **`demo_build/` is git-ignored but contains `vo/*.wav` and a smoke PNG.**
   Harmless, but confirm no stray artefact gets force-added later.

## MEDIUM
8. Commit `4b72a55 "heat-vision: HackaLaunch submission"` sits mid-history
   between feature commits and a later `refactor:` commit — sloppy, invites a
   judge to wonder what was squashed. Reword or reorder before push.
9. README "Honest limitations" says "The numbers below are real measurements"
   while pointing at an empty section — reword once metrics land.
10. Demo recorder's primary TTS path is Windows `System.Speech`; nothing in the
    repo proves a Linux judge gets audio. `--no-audio` fallback is documented,
    which is enough, but say so in README, not only RUN.md.

## Concrete fix list (ordered)
1. Let training finish; run `scripts/evaluate.py --checkpoint runs/fused/heatnet_s.pt --test 400` and `--ablations`.
2. Un-ignore `results/` (or `git add -f` the three result files) and run `scripts/update_readme.py` so METRICS/ABLATION/TESTS blocks are filled with precision, recall and a missed-hazard count.
3. Record `demo.mp4` (≤3:00), upload to YouTube unlisted or a GitHub release, replace `<DEMO_VIDEO_URL>`.
4. Publish the checkpoint externally (HF or release asset) and link it in README.
5. Commit all four dirty files, tidy the commit message ordering, then push to the ONE canonical repo and make `RUN.md`'s `gh` line match it exactly:
   `gh repo create valeemlbb-cell/hackalaunch-heat-vision --public --source=. --push` (or rename the remote and use the warung-ops name — pick one).
6. Re-run `python -m pytest -q` from a fresh clone before submitting.

With items 1–5 done this is a 88–92 entry: the safety-case write-up, the
adversarial subsets and the autonomy disclosure are genuinely above the field.
