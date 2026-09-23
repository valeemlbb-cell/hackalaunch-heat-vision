<!-- generated-header v2 -->
# SUBMISSION — heat-vision

Paste-ready. Five fields, in the order the HackaLaunch form asks for them.
Refreshed 2026-09-24T05:06:10+07:00.

---

## TITLE  (71/80 chars)

```
HeatVision — bispectral detection that pulls lithium cells off the belt
```

## DESCRIPTION  (1476 chars)

```
HeatVision is a complete sorting station for recycling lines: find the lithium cell in the waste stream and physically remove it before the crusher reaches it.

What it does: colour plus long-wave infrared frames go into a fused detector, objects are tracked down the belt, a safety policy decides what is dangerous, and a 4-DOF arm removes what it condemns — then checks whether the pick succeeded.

How it works: one camera is not enough in either direction. Thermal alone false-alarms on brake discs, motor fragments and anything sun-warmed; the line stops all day and somebody switches the system off. Colour alone misses the discharged cell sitting at ambient that looks like a black plastic shard. HeatVision fuses both at the first convolution but keeps them separately accountable: vision decides what an object is, thermal how dangerous it is right now, and a rule set combines them. Every decision is logged with the rule id and the numbers behind it. Both failure modes are named adversarial subsets in the dataset and reported as first-class metrics, including hazards missed on a held-out test set.

Real vs mocked: simulation-only, which the brief allows — the detector, policy, tracker, metrics and arm control loop are real code and real evaluation; the belt, sensors and arm are simulated. No physical hardware is behind this entry.

How to run: `pip install -r requirements.txt && pytest -q`, then the pipeline command in RUN.md. Laptop CPU, no GPU, no keys.
```

## REPO URL

```
https://github.com/valeemlbb-cell/hackalaunch-heat-vision
```

## VIDEO URL

```
VIDEO_URL_PENDING
```

> The main session posts `demo_x.mp4` from this folder to X and replaces the
> line above with the public post URL. The form needs a **link**; a file is useless.

## SOLANA PAYOUT ADDRESS

```
7W31iaCmjerN1jkpEnmZevn74SZxv83yEQvLsnc4PS7Q
```

---

## Appendix — earlier submission notes (kept verbatim)

# SUBMISSION — Heat Vision (hackalaunch.com/h/heat-vision)

Operator sheet. The paste-ready prose for the form's description box lives in
[`DESCRIPTION.md`](DESCRIPTION.md); this file is the checklist around it.

## Form fields

| Field | Value |
|---|---|
| Project name | **HeatVision** |
| Public GitHub repository | `https://github.com/valeemlbb-cell/hackalaunch-heat-vision` (branch `main`) |
| Demo video (public link) | `<VIDEO_URL>` — upload `demo.mp4` to YouTube (unlisted) or a GitHub release, then paste the link |
| Short description | the whole block in `DESCRIPTION.md`, after replacing the two bracketed URLs |
| Solana payout address | `7W31iaCmjerN1jkpEnmZevn74SZxv83yEQvLsnc4PS7Q` |
| Team | Warung Ops — Henggar · X [@issue0x](https://x.com/issue0x) · Telegram @sambobolo · rakavaleeqa@warungsosmed.store |

Local artefacts: `demo.mp4` (1080p, 25 fps, under the 3:00 cap, git-ignored —
upload it, do not commit it), `results/metrics.json`, `results/METRICS.md`,
`results/ablations.json`, `weights/heatnet_s.pt` (5.2 MB, committed).

## Before you submit

Human steps only — the build agent does not push, does not upload and does not
submit. Full commands are in [`RUN.md`](RUN.md).

1. `git status --porcelain` prints nothing, and
   `git rev-list --left-right --count origin/main...HEAD` prints `0	0`.
   A judge must clone exactly the tree that was tested.
2. **Delete the stale `master` branch on the remote** and set the default
   branch to `main`. The remote still carries an early `master` pointing at a
   mid-history commit with no metrics and no weights; anyone landing there
   sees a repo that looks unfinished. `git push origin --delete master`.
3. Upload `demo.mp4`, then replace `<DEMO_VIDEO_URL>` in `README.md` and
   `<REPO_URL>` / `<VIDEO_URL>` in `DESCRIPTION.md`, commit and push.
4. Re-run `python -m pytest -q` from a fresh clone of the public repo.
5. Confirm the video runs under 3:00 and that the stress-test hazard-rate
   banner is on screen throughout.

## What the rules ask for, point by point

| Requirement | How it is met |
|---|---|
| Public GitHub repository | `valeemlbb-cell/hackalaunch-heat-vision`, MIT licence, README at root |
| Demo video ≤ 3 minutes | 1080p/25fps, title card + live closed-loop run + results card, offline voice-over |
| Description | `DESCRIPTION.md` |
| Detection metrics on held-out data | `results/METRICS.md`, regenerated into the README by `scripts/update_readme.py`: mAP@0.5 **0.965**, precision **0.956**, recall **0.973** on 400 seeds the model never trained on |
| Hazards-missed count | reported as an absolute number, not a rate: the end-to-end table gives hazards presented, removed into a bin, and how many reached the crusher |
| Adversarial honesty | named subsets: 91.5% recall on discharged cells at ambient (invisible to thermal), 9.9% false-alarm on hot non-battery decoys (what breaks a thermal-only trigger) |
| Ablation | same architecture, same epochs, same frames, only the input channels differ — `fused` vs `rgb` (thermal blanked) vs `lwir` (colour blanked), in `results/ablations.json` and the README |
| Not detection-only | belt halt, 4-DOF arm pick, gripper force selection, and a **verified** grasp with an explicit `grasp_miss` path — misses are counted, not hidden |
| Autonomy disclosure | `docs/AUTONOMY.md` answers the disqualification clause by name: what is computed live, what is scripted (narration timeline, title/results cards), and that the recording uses an inflated `--hazard-rate 0.5` which is also shown on screen |
| Pre-hackathon work | none. Every file written during the window from an empty directory; stated in the README and `docs/AUTONOMY.md` |
| Safety / no real cells | 100% synthetic simulator. No physical lithium cell is touched, heated or punctured. Ground-truth access is confined to `pipeline.py`, every such site is marked SIMULATION ONLY, and it is invisible to the detector, the tracker and the policy |
| Dataset licensing | 100% self-generated and seed-addressed; no external dataset, so no dataset licence is implicated |
| Asset licensing | MIT, no bundled fonts, images or audio — text is OpenCV's built-in Hershey vector fonts, narration is synthesised offline at build time |
| Large weights available | `weights/heatnet_s.pt` is committed (5.2 MB), so evaluate and record both work straight from a clone with no retrain |
| No secrets | none exist. `.env` git-ignored, `.env.example` carries only optional hardware/TTS variables. `git log -p \| grep -iE "api[_-]?key\|secret\|token\|BEGIN .*PRIVATE"` finds only prose |
| No admin backdoor | no network listener, no remote control, no override path |
| No mainnet / real money | no chain code at all; the payout address appears only in operator notes, never in shipped code |
| AI disclosure | stated in the README and in `DESCRIPTION.md`: built with an AI coding agent under human direction, no AI service called at runtime |

## Known limitations we state rather than hide

* Synthetic data only. The numbers say the architecture and the decision layer
  work; they say nothing about transfer to real footage.
* The arm is kinematic, not dynamic — grasp success is a positional tolerance,
  not a contact simulation.
* One arm, so throughput (not detection) is the main reason hazards reach the
  crusher in the end-to-end trials. The `too_late_flags` counter makes that
  visible.
* The thermal model is physically motivated but not calibrated against real
  cells.
