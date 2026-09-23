# REVIEW_3 — token-holder judge pass

**Packet:** heat-vision · **Reviewed:** 2026-09-24 ~03:35 WIB · **Reviewer:** Judge 3 (voting token holder)
**Rules re-read:** https://hackalaunch.com/h/heat-vision (submissions close Sep 28 13:15 UTC; 24h token-holder vote; pool ~11.6 SOL + all future fees)

**Score: 68 / 100** — as the packet stands *right now*. The engineering is top-quartile; the
things a voter actually looks at do not exist yet. Ceiling is ~88 if the fixes below land.

---

## 1. Rules compliance

| Rule requirement | Status |
| --- | --- |
| see → decide → pick, all three | PASS — `detect/` + `decide/policy.py` (R0–R7) + `arm/` with grasp verification. Not a detection-only entry. |
| Public GitHub repo, functioning README | **BLOCKED** — repo is local only; `gh repo create` line is in RUN.md as required by our own agent limits, so this is a human step, not a defect. |
| Model results: dataset sources, held-out test set, precision/recall, **"how many hazards it missed"** | **FAIL as of now.** README has `<!-- METRICS:BEGIN -->` / `ABLATION` / `TESTS` placeholders with zero numbers. No `results/` directory. `runs/` has only `fused` and the training log stops at epoch 11/14. The two ablation variants (`rgb`, `lwir`) have not been trained at all, yet the README promises a table comparing them. |
| Demo video ≤3 min showing detection, decision, physical removal | **FAIL as of now.** No `demo.mp4`. `demo_build/` holds 7 VO wavs and one smoke PNG. README link is literally `<DEMO_VIDEO_URL>`. |
| Project description / rationale | PASS — strongest part of the packet. The "thermal alone false-alarms, colour alone misses the discharged half" framing is a real thesis, not a feature list. |
| No API keys, `.env.example` | PASS — `.env.example` is documentation-only, pipeline needs no secrets, `.env` git-ignored. |
| Pre-existing work marked | PASS — explicit "Pre-hackathon work: none" plus `docs/AUTONOMY.md`, git history starts at scaffold commit. |
| Never damage live lithium cells | PASS — simulation only, stated explicitly. |
| Dataset licence | PASS — 100% self-generated, seed-addressed, disjoint splits asserted by a test. Best-in-class answer to this rule. |
| Disqualifier: "pre-scripted or teleoperated presented as autonomous" | **AT RISK.** `docs/AUTONOMY.md` and the README disclose that the narration timeline, the title/results cards and an elevated hazard rate are scripted. That disclosure lives in the repo — the *video* is where a hostile voter will look. If the elevated hazard rate is not on screen, someone will call this staged and the DQ clause is explicitly worded to catch it. |

## 2. Would I vote for this over a typical submission?

Right now: **no**, because I cannot. There is nothing to watch and no number to quote.
A token holder skims a video, skims a README hero image, and votes. This packet currently
offers neither — and even the README's own results section tells the reader to go run a
script. In a 24-hour vote that reads as "unfinished", which is far worse than "modest".

If the video and the metrics land, then **yes**, and comfortably. Three things separate it
from the typical entry:

- It has a **failure thesis**, not just a model. The hot-decoy / cold-battery adversarial
  subsets are evaluated as first-class metrics, so the entry can say "here is what breaks
  the obvious approach and here is my number on it". Almost nobody else will do this.
- **The safety policy is a legible artifact.** Eight numbered rules, a rule id logged with
  every decision, and "a cell in thermal runaway is never gripped" as an explicit
  guarantee with a test pinning it. That is the detail that makes an industrial judge
  believe a human thought about this.
- **Honest limitations section** that names the kinematic arm, synthetic-only transfer,
  and the throughput bottleneck (`too_late_flags`). Voters reward this more than authors expect.

Against it: it is simulation-only in a field where one shaky phone video of a real arm
flicking a vape off a belt beats a beautiful simulator. That is the structural risk and
it cannot be fixed by Sep 28 — it can only be out-presented.

## 3. The single change that most raises its odds

**Put a moving picture at the top of the README, and put the headline hazard number
next to it — both before anyone clicks the video link.**

Concretely: a ~6 s looping GIF of the operator console (detection boxes + thermal panel +
the arm actually taking an item off the belt), embedded at line ~5 of README.md, with one
line under it: *"N of M hazards removed before the crusher; K missed; hot-decoy false-alarm
rate X%."* Everything else in this packet is already better than it needs to be. The gap is
that a voter with 40 seconds currently sees a wall of prose and three empty metric blocks.

## 4. Concrete fixes, ordered

1. **Finish the fused run and fill the three README blocks.** Placeholders in a submitted
   README are a self-inflicted wound; a voter reads "Run `python scripts/evaluate.py` to
   regenerate this section" as "the author did not have numbers".
2. **Train the two ablations or delete the ablation claim.** The README asserts a three-way
   comparison ("same architecture, same epochs, only the input channels differ"). If only
   `fused` exists at submission, that paragraph is an unbacked claim in a contest whose DQ
   list starts with plagiarism and non-functional demos. If time is short, cut the epoch
   count for the ablations and say so, or drop the section entirely.
3. **Record `demo.mp4` and replace `<DEMO_VIDEO_URL>`.** A submitted README containing a
   literal angle-bracket placeholder is the single most damaging cosmetic defect available.
4. **Add the README hero GIF + headline numbers** (see §3). `demo_build/_smoke_console.png`
   already proves the console renders — it is not referenced anywhere in the README.
5. **Burn the autonomy disclosure into the video itself**: one caption, ~2 s, "hazard rate
   raised for the clip; detection, decision, trajectory and grasp are computed live; no
   teleoperation." Protects against the DQ clause where it actually gets applied.
6. **Surface "hazards missed" as a named headline stat**, not a row inside a table. The
   rules ask for it by name (`too_late_flags` + any undetected hazard); give it its own
   bolded line so a judge checking the rubric can tick it in two seconds.
7. **Lead the README with a one-sentence "what you are looking at"** above the current
   problem essay. The prose is excellent but the first screen is all text; a voter should
   know it is a working sim-sorter before paragraph three.
8. **Housekeeping before publish:** `.pytest_cache/`, `__pycache__/`, `data/cache/*.npz`
   (two cached arrays are on disk) and `runs/*.pt` must be git-ignored and absent from the
   pushed tree — verify with `git status --ignored` and the credential grep already in
   RUN.md's checklist.
9. **State the test count in the README** rather than leaving `TESTS:BEGIN` empty. "151
   tests" is in the commit message and is a credibility number; it should be visible.

## 5. Notes for the other judges

- Do not re-litigate the simulation-only choice; the rules permit it explicitly and the
  rationale in "Hardware and simulator" is sound.
- The `.env.example` is clean — no live values, and the pipeline genuinely needs no secrets.
- Nothing in this packet needs an account, a wallet connection or an outbound post.
