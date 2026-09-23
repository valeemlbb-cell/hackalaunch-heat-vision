# HeatVision

**Find the lithium in the waste stream and get it off the belt before the crusher does.**

A complete sorting station: a bispectral (colour + long-wave infrared) detector,
an explicit safety policy, and a 4-DOF arm that physically removes what the
policy condemns — then checks whether it actually got it.

Built for the [Heat Vision](https://hackalaunch.com/h/heat-vision) hackathon.
Simulation-only entry, which the brief allows. Runs end to end on a laptop CPU
with no GPU, no cloud service and no API keys.

**Demo video (≤3 min):** <DEMO_VIDEO_URL>

---

## The problem, and why one camera is not enough

A lithium cell that goes into a shredder is a fire. Recyclers already know
this; the hard part is that neither obvious sensor works on its own.

**A thermal camera alone false-alarms constantly.** Scrap is full of hot things
that are not batteries: a brake disc off a car, a motor fragment, anything that
has been in the sun. If heat alone stops the line, the line stops all day, and
within a week somebody turns the system off. That is the real failure mode.

**A colour camera alone misses the dangerous half.** A discharged cell sits at
ambient temperature and looks exactly like a black plastic shard. A swollen
pouch inside a crushed phone looks like a crushed phone.

HeatVision fuses both at the first convolution and then — critically — keeps
the two signals *separately accountable* in the decision logic. Vision decides
**what** an object is. Thermal decides **how dangerous it is right now**. An
explicit rule set combines them, and every decision is logged with the rule id
and the numbers that fired it.

Both failure modes above are built into the dataset as named adversarial
subsets, and both are reported as first-class metrics below.

## What it does, end to end

```
colour + LWIR frame  ->  HeatNet-S  ->  thermal measurement  ->  tracking
                                                                    |
                                                                    v
                                                            safety policy
                                                          (R0..R7, logged)
                                                                    |
            +---------------------------+-----------------------+---+
            v                           v                       v
     EMERGENCY_STOP              EXTRACT_QUENCH        EXTRACT_QUARANTINE
     belt halt, human            belt stopped,         pick on the fly,
     callout, NO grip            5 N gentle grip       12 N grip
                                        |                       |
                                        v                       v
                                   sand bin              quarantine bin
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module map and
[`docs/SAFETY.md`](docs/SAFETY.md) for the full rule table and safety case.

## Results

<!-- METRICS:BEGIN -->
Run `python scripts/evaluate.py` to regenerate this section.
<!-- METRICS:END -->

### What the ablation says

<!-- ABLATION:BEGIN -->
Run `python scripts/evaluate.py --ablations` to regenerate this section.
<!-- ABLATION:END -->

All three variants are the same architecture trained for the same number of
epochs on the same frames; only the input channels differ. `rgb` has the
thermal channel blanked, `lwir` has the colour channels blanked.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                    # the whole suite
python scripts/train.py --train 3000 --val 400 --epochs 14
python scripts/evaluate.py --checkpoint runs/fused/heatnet_s.pt --test 400
python scripts/record_demo.py --seconds 88 --out demo.mp4
```

Full commands, timings and the publishing steps are in [`RUN.md`](RUN.md).

## Hardware and simulator

There is **no physical hardware in this project**. Everything runs against a
simulator written for this repository:

* **Sensors.** A procedural waste-stream renderer produces registered RGB and
  LWIR frame pairs at 256×256. The thermal channel is a radiometric model —
  `apparent = ε·T_object + (1−ε)·T_reflected` — with a distance-transform core
  profile, thermal bleed into the belt, Gaussian optics blur, NETD noise
  (0.055 K, FLIR Lepton class) and per-column fixed-pattern noise. Low
  emissivity therefore genuinely hides a hot core, which is the physical reason
  a thermal-only trigger cannot be trusted.
* **Robot.** A SCARA-style 3R arm plus a vertical axis and a two-finger
  gripper, with analytic IK, both elbow solutions, workspace limits and
  trapezoidal joint trajectories under published-envelope velocity and
  acceleration limits. It is a **kinematic** model, not a dynamic one: no
  contact physics, no friction. Grasp success is decided by a 26 mm positional
  tolerance at the instant the jaws close.
* **No real lithium cells were used, heated, punctured or damaged.** Nothing in
  this project touches physical batteries.

Why not PyBullet or Isaac: the point of this entry is the *decision layer* and
the *sensor fusion*, and a purpose-built kinematic cell keeps the whole loop
deterministic, seed-reproducible, dependency-light and fast enough to score
hundreds of trials on a CPU. The arm interface (`Arm.start` / `Arm.step` /
`PickResult`) is deliberately narrow so a real controller or a physics
simulator can be dropped in behind it.

## Training methodology

* **Model.** HeatNet-S, 1,282,695 parameters. Four input channels (R, G, B,
  calibrated LWIR) fused at the stem, an encoder down to stride 16, a
  lightweight decoder back to stride 4, and a CenterNet head (per-class
  heatmap + size + sub-cell offset). Anchor-free, single-shot.
* **Losses.** Penalty-reduced focal loss on the heatmap (α=2, β=4), masked L1
  on size and offset. The heatmap bias is initialised to −4 so training starts
  from a sane background prior.
* **Schedule.** AdamW, lr 2e-3, cosine decay with 60 warm-up steps, weight
  decay 1e-4, gradient clipping at 5.0, batch 16, 14 epochs on 3,000 frames.
  CPU only.
* **Augmentation.** Belt-plane flips (a sorter can run either way round) and a
  small thermal gain/offset jitter modelling sensor drift and ambient change.
  Nothing that would break RGB↔LWIR registration.
* **Model selection.** Best validation loss across epochs; validation frames
  come from their own seed range.

## Dataset

**100% synthetic, generated by this repository. No external dataset was
downloaded or used, so no dataset licence is implicated.**

The dataset is *seed-addressed*: a split is a list of integer seeds, and
`render_sample(seed)` is deterministic, so `data/manifest.json` (a few hundred
bytes) fully defines it and anybody can regenerate it byte-for-byte.

| Split | Frames | Seed range |
| --- | --- | --- |
| train | 3,000 | 1,000,000 – 1,002,999 |
| val | 400 | 5,000,000 – 5,000,399 |
| test (held out) | 400 | 9,000,000 – 9,000,399 |

The three ranges are disjoint by construction and
`tests/test_metrics.py::TestSplitHygiene` asserts it. `split_digest()` gives a
SHA-256 over a rendered split, printed by `scripts/evaluate.py` into
`results/metrics.json`, so the test set cannot quietly move between runs.

### Classes and the adversarial subsets

Detector classes: `cell` (cylindrical 18650 / AA), `pouch` (LiPo, possibly
swollen), `device` (phone, vape, power bank with an embedded cell).

Everything else is background — cardboard, PET, aluminium, PCB scrap, fabric,
foil — plus four named subsets that exist to make the evaluation honest:

| Subset | What it is | What it breaks |
| --- | --- | --- |
| `cold_battery` | discharged cell at ambient | thermal-only detection |
| `thermal_event` | swollen pouch, venting, 34–72 K and rising | anything that only classifies |
| `hot_decoy` | brake disc / motor fragment, 18–58 K, not a battery | thermal-only triggering |
| `cell_lookalike` | black plastic shard, shiny bolt | colour-only detection |

## Repository layout

```
heatvision/
  config.py          every constant and threshold, one file
  sim/               materials, scene, renderer, seed-addressed dataset
  detect/            HeatNet-S, target encoding, decoding, training, inference
  decide/            thermal measurement, belt tracking, the safety policy
  arm/               kinematics, trajectories, the pick state machine
  viz/               operator console and demo cards
  evaluation/        AP / mAP, adversarial subsets, end-to-end trials
  pipeline.py        the closed loop
scripts/             make_dataset, train, evaluate, record_demo
tests/               the suite
docs/                ARCHITECTURE, SAFETY, AUTONOMY
```

## Tests

<!-- TESTS:BEGIN -->
Run `python -m pytest -q` to regenerate this section.
<!-- TESTS:END -->

The suite covers the radiometric model (a low-emissivity surface must hide its
heat; a discharged cell must be invisible to thermal), every policy rule
including the hot-decoy case, IK/FK round trips and trajectory limits, the full
pick cycle including a deliberately missed grasp, the metric implementations
against hand-computed values, and the closed loop's safety behaviour.

## Disclosures

**Autonomy.** Everything between the camera frame and the bin is computed live:
detection, thermal measurement, tracking, the rule that fires, the intercept,
the trajectory and the grasp check. There is no scripted event list and no
teleoperation. The narration timeline and the title/results cards in the video
are scripted, and the recording uses a higher hazard rate than a real stream so
a 90-second clip is worth watching. Full detail in
[`docs/AUTONOMY.md`](docs/AUTONOMY.md).

**Pre-existing work: none.** Every file here was written for this hackathon
from an empty directory. No code, assets or models were carried in from earlier
projects.

**AI agent usage.** Built with heavy use of an AI coding agent (Claude, via
Claude Code) under human direction: the agent wrote most of the implementation,
tests and docs; a human set the framing, the safety rules and the architecture
decisions, and reviewed the output. No AI service is called at runtime.

**Secrets.** There are none. The pipeline needs no credentials at all;
`.env.example` documents the optional variables you would need only when
pointing this at real cameras and a real controller, and `.env` is git-ignored.

**Licence.** MIT, see [`LICENSE`](LICENSE). No bundled fonts, images or audio;
all text rendering uses OpenCV's built-in Hershey vector fonts.

## Honest limitations

* Synthetic data only. The numbers below are real measurements on held-out
  synthetic seeds; they say the architecture and the decision layer work, and
  they say nothing about how the network transfers to real footage. Anyone
  deploying this would fine-tune on annotated real frames first.
* Kinematic arm, not dynamic. Grasp success is a positional tolerance, not a
  contact simulation.
* One arm. Throughput, not detection, is the main cause of hazards reaching the
  crusher in the end-to-end trials — the `too_late_flags` counter makes that
  visible rather than hiding it.
* The thermal model is physically motivated but not calibrated against real
  cells.
