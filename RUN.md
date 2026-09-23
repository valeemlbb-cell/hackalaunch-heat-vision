# Operator runbook

Everything below has been run on this machine (Windows 11, Python 3.14, CPU
only). Times are from that run.

## 0. Environment

```bash
python -m venv .venv && .venv/Scripts/activate   # or: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional; nothing in the pipeline needs secrets
```

`ffmpeg` must be on `PATH` only if you want to re-record the demo video.

## 1. Tests

```bash
python -m pytest -q
python -m pytest --cov=heatvision --cov-report=term-missing
```

## 2. Dataset

```bash
python scripts/make_dataset.py --train 3000 --val 400 --test 400
python scripts/make_dataset.py --test 400 --digest   # prints the split SHA-256
```

Writes `data/manifest.json`. Frames are rendered on demand from their seeds; the
first training run caches them under `data/cache/` (git-ignored).

## 3. Train

```bash
TRAIN="--train 3000 --val 400 --epochs 14"
python scripts/train.py $TRAIN                                        # fused
python scripts/train.py $TRAIN --ablation rgb  --out runs/rgb         # no thermal
python scripts/train.py $TRAIN --ablation lwir --out runs/lwir        # no colour
```

About 20 minutes per variant on six CPU threads (longer if you run two at
once). Checkpoints land in `runs/<variant>/heatnet_s.pt`.

Pass the same `$TRAIN` flags to all three. The ablation table only means
something if the variants got an identical budget, and the script's own
defaults are smaller (2400/300/12) than what was shipped. The runs behind the
committed `results/ablations.json` reached best validation loss 0.1827 (fused),
0.2091 (rgb) and 0.2939 (lwir).

## 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/fused/heatnet_s.pt --test 400
python scripts/evaluate.py --ablations --test 400
```

Writes `results/metrics.json`, `results/METRICS.md` and `results/ablations.json`,
all three of which are committed. `python scripts/update_readme.py` then copies
them into the README between its marker comments, so the README tables cannot
drift from what the evaluator actually produced.

**A fresh clone can evaluate without retraining.** The shipped fused checkpoint
is committed at `weights/heatnet_s.pt` (5.2 MB) and is what
`default_checkpoint()` picks up, so `scripts/evaluate.py` and
`scripts/record_demo.py` both work immediately after `git clone`:

```bash
python scripts/evaluate.py --checkpoint weights/heatnet_s.pt --test 400
```

Only the two ablation variants (`runs/rgb`, `runs/lwir`) need the ~20 min
training run above; their measured numbers are already in
`results/ablations.json` and in the README table.

## 5. Demo video

```bash
python scripts/record_demo.py --seconds 88 --out demo.mp4
python scripts/record_demo.py --seconds 88 --out demo.mp4 --no-audio   # silent
```

Produces a 1080p/25fps file under the 3-minute limit (title card, 88 s of live
run, results card). Intermediates land in `demo_build/` (git-ignored).

The voice-over is synthesised offline by the local Windows voice
(`System.Speech`); no account, no key and no network are needed. On a machine
without it the recorder falls back to `edge-tts`, and failing that it writes the
video silently with the on-screen captions intact.

---

# Publishing this repository

The agent never pushes and never submits. Both are human steps.

The canonical remote already exists:

```
https://github.com/valeemlbb-cell/hackalaunch-heat-vision
```

From the repo root, push the finished work to it:

```bash
gh auth login                    # once, if needed
git push -u origin HEAD:main     # or: gh repo create valeemlbb-cell/hackalaunch-heat-vision --public --source=. --push
```

Use that one repository for the submission. Do not create a second one under a
different owner — two repos is the fastest way to submit a URL that 404s.

**Delete the stale `master` branch on the remote before submitting.** The
remote currently carries two branches: `main` (current, what the submission
points at) and `master`, left over from an early push and stuck at an old
mid-history commit. A judge who lands on `master` clones a tree that has no
metrics, no weights and no results. Fix it once:

```bash
git ls-remote --heads origin              # expect main AND master right now
git push origin --delete master           # leave only main
gh repo edit valeemlbb-cell/hackalaunch-heat-vision --default-branch main
git ls-remote --heads origin              # expect main only
```

Then confirm the public tree is exactly the tested tree:

```bash
git status --porcelain                    # must print nothing
git rev-list --left-right --count origin/main...HEAD   # must print "0<TAB>0"
```

The trained checkpoint (`weights/heatnet_s.pt`, 5 MB) is committed, so a judge
can clone and run `scripts/evaluate.py` immediately without a 20-minute
retrain. `demo.mp4` is NOT committed:

1. upload `demo.mp4` to YouTube (unlisted is fine) or attach it to a GitHub
   release;
2. replace `<DEMO_VIDEO_URL>` in `README.md` and the two bracketed URLs in
   `DESCRIPTION.md`;
3. `git commit -am "docs: link the demo video" && git push`.

## Submission checklist

- [ ] repo public, README renders, tests green in a fresh clone
- [ ] `results/METRICS.md` committed and matching the README tables
- [ ] `weights/heatnet_s.pt` committed (so the metrics are reproducible)
- [ ] `demo.mp4` uploaded, README + DESCRIPTION links replaced, video under 3:00
- [ ] the video shows the stress-test hazard-rate banner on screen throughout
- [ ] `docs/AUTONOMY.md` linked from the README (autonomy + pre-hackathon work
      disclosure required by the rules)
- [ ] no `.env`, no keys, no credentials anywhere in the history:
      `git log -p | grep -iE "api[_-]?key|secret|token|BEGIN .*PRIVATE"`
- [ ] paste `DESCRIPTION.md` into the submission form
- [ ] payout address on the form:
      `7W31iaCmjerN1jkpEnmZevn74SZxv83yEQvLsnc4PS7Q`
