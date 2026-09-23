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
python scripts/train.py --train 3000 --val 400 --epochs 14            # fused
python scripts/train.py --ablation rgb  --out runs/rgb                # no thermal
python scripts/train.py --ablation lwir --out runs/lwir               # no colour
```

About 20 minutes per variant on six CPU threads. Checkpoints land in
`runs/<variant>/heatnet_s.pt`.

## 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/fused/heatnet_s.pt --test 400
python scripts/evaluate.py --ablations --test 400
```

Writes `results/metrics.json`, `results/METRICS.md` and `results/ablations.json`.
The README tables are copied verbatim from those files.

## 5. Demo video

```bash
python scripts/record_demo.py --seconds 88 --out demo.mp4
python scripts/record_demo.py --seconds 88 --out demo.mp4 --no-audio   # silent
```

Produces a 1080p/25fps file under the 3-minute limit. Intermediates land in
`demo_build/` (git-ignored).

---

# Publishing this repository

`gh` is not authenticated in the build environment, so the repository is
prepared locally with git history intact and pushed by a human. From the repo
root:

```bash
gh auth login                       # once, if needed
gh repo create warung-ops/heatvision --public --source=. --push
```

Then, because the demo video is larger than GitHub's comfortable file size and
is git-ignored:

1. upload `demo.mp4` to YouTube (unlisted is fine) or attach it to a GitHub
   release;
2. replace `<DEMO_VIDEO_URL>` in `README.md` with the link;
3. `git commit -am "docs: link the demo video" && git push`.

## Submission checklist

- [ ] repo public, README renders, tests green in a fresh clone
- [ ] `results/METRICS.md` present and matching the README tables
- [ ] `demo.mp4` uploaded, README link replaced, video under 3:00
- [ ] `docs/AUTONOMY.md` linked from the README (autonomy + pre-existing work
      disclosure required by the rules)
- [ ] no `.env`, no keys, no credentials anywhere in the history:
      `git log -p | grep -iE "api[_-]?key|secret|token|BEGIN .*PRIVATE"`
- [ ] payout address on the submission form:
      `7W31iaCmjerN1jkpEnmZevn74SZxv83yEQvLsnc4PS7Q`
