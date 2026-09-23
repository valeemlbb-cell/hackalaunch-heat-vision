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
### Detection (held-out test split)

- frames: 400  |  operating threshold: 0.3
- **mAP@0.5: 0.965**  |  mAP@[0.5:0.95]: 0.601
- inference: 11.3 ms/frame (CPU)

| class | AP@0.5 | precision | recall | GT |
| --- | --- | --- | --- | --- |
| cell | 0.926 | 0.944 | 0.949 | 624 |
| pouch | 0.976 | 0.924 | 0.980 | 349 |
| device | 0.994 | 0.990 | 0.995 | 582 |
| **any lithium** | - | **0.956** | **0.973** | 1555 |

### Adversarial subsets

| subset | n | metric | value | what it tests |
| --- | --- | --- | --- | --- |
| `cold_battery` | 189 | recall | **0.915** | discharged cell at ambient - thermal is blind to it |
| `thermal_event` | 148 | recall | **0.953** | swollen pouch, venting |
| `ordinary` | 1218 | recall | **0.984** | every other lithium item |
| `hot_decoy` | 282 | false alarm | **0.099** | hot brake disc / motor fragment, NOT a battery |
| `cell_lookalike` | 485 | false alarm | **0.027** | black plastic shard, shiny steel bolt |
| `ordinary` | 1490 | false alarm | **0.002** | ordinary clutter |

### End-to-end removal (closed loop, scenes the model never saw)

6 runs x 45 s of belt time per scenario, fresh seeds, full loop: detect -> decide -> intercept -> grip -> verify.

| scenario | hazards | robot removed | human removed | kept from crusher | **missed** | `too_late` | false picks | e-stops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stress mix (28% hazards) | 104 | **62 (60%)** | 7 | **66%** | **28 (27%)** | 42 | 3 | 10 |
| realistic mix (6% hazards) | 23 | **21 (91%)** | 1 | **96%** | **1 (4%)** | 18 | 17 | 3 |

Mean pick cycle 1.7752 s, pick success 0.82.
Under the stress mix almost every miss is a `too_late` flag: the item was seen and classified correctly, but the single arm was still finishing an earlier pick. That is a throughput limit, not a perception one, and a second arm or a downstream diverter fixes it. The realistic-mix row is the same code with the hazard share a real waste stream would present.

Held-out test split SHA-256: `2d3e04c4abb534c42249c59b2e520d36...`

<!-- METRICS:END -->

### What the ablation says

<!-- ABLATION:BEGIN -->
| variant | mAP@0.5 | precision | recall | cold-battery recall | hot-decoy false alarm | lookalike false alarm |
| --- | --- | --- | --- | --- | --- | --- |
| **RGB + LWIR (shipped)** | 0.965 | 0.956 | 0.973 | 0.915 | 0.099 | 0.027 |
| RGB only | 0.945 | 0.923 | 0.960 | 0.905 | 0.156 | 0.047 |
| LWIR only | 0.943 | 0.805 | 0.955 | 0.720 | 0.206 | 0.472 |
<!-- ABLATION:END -->

All three variants are the same architecture trained for the same number of
epochs (14) on the same frames; only the input channels differ. `rgb` has the
thermal channel blanked, `lwir` has the colour channels blanked.

Read that table honestly: **mAP barely separates the three.** On the headline
number RGB-only is 0.945 against the fused 0.965, and a reviewer would be right
to say that alone does not justify a second camera. The separation is in the
columns that correspond to the two failure modes this project exists to fix:

* **Thermal-only cannot see a discharged cell.** `lwir` drops to 0.720 recall
  on `cold_battery` against 0.915 fused — roughly one in four flat cells walks
  into the crusher.
* **Thermal-only is fooled by anything warm or shiny.** Its false-alarm rate on
  `cell_lookalike` is 0.472 against 0.027 fused: seventeen times as many
  needless stops, which is precisely the failure that gets a real system
  switched off.
* **Colour-only false-alarms on hot decoys** at 0.156 against 0.099 fused, and
  gives up precision overall (0.923 against 0.956).

So the fused model is not dramatically better at *finding boxes*; it is
markedly better at **not being wrong in the two specific ways that matter on a
sort line**. That is the claim this repository makes, and the table is the
evidence for it.

## Quick start

The trained fused checkpoint is committed at `weights/heatnet_s.pt` (5.2 MB),
so a fresh clone can reproduce the numbers above and record the video **without
retraining**:

```bash
pip install -r requirements.txt
python -m pytest -q                                              # the whole suite
python scripts/evaluate.py --checkpoint weights/heatnet_s.pt --test 400
python scripts/record_demo.py --seconds 88 --out demo.mp4
python scripts/record_demo.py --seconds 88 --out demo.mp4 --no-audio   # silent
```

Retraining from scratch is optional and takes about 20 minutes per variant on
six CPU threads:

```bash
TRAIN="--train 3000 --val 400 --epochs 14"       # identical budget for all three
python scripts/train.py $TRAIN                           # fused
python scripts/train.py $TRAIN --ablation rgb  --out runs/rgb    # no thermal
python scripts/train.py $TRAIN --ablation lwir --out runs/lwir   # no colour
python scripts/evaluate.py --ablations --test 400        # the table above
```

The three variants must get the same budget or the comparison is meaningless,
which is why the flags are shared rather than left to the defaults.

`ffmpeg` is needed only for the video. The voice-over is synthesised offline by
the local system voice (Windows `System.Speech`), falling back to `edge-tts`;
on a machine with neither, `--no-audio` writes the video silently with the
on-screen captions intact. No account, key or network is required either way.

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
```
$ python -m pytest -q
171 passed in 57.84s
```

all passing.
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

**Pre-hackathon work: none.** Every file in this repository was written during
the hackathon window, from an empty directory. Nothing here was built before
the hackathon: no prior work, no code, assets, datasets or trained weights were
carried in from earlier projects, and the git history starts at the first
commit of this entry. The only pre-existing components are third-party
dependencies — NumPy, PyTorch, OpenCV, pytest, and optionally `edge-tts` and
ffmpeg for the demo audio — used as libraries under their own licences. See
[`docs/AUTONOMY.md`](docs/AUTONOMY.md#pre-hackathon-work).

**AI agent usage.** Built with heavy use of an AI coding agent (Claude, via
Claude Code) under human direction: the agent wrote most of the implementation,
tests and docs; a human set the framing, the safety rules and the architecture
decisions, and reviewed the output. No AI service is called at runtime.

**Secrets.** There are none. The pipeline needs no credentials at all;
`.env.example` documents the optional variables you would need only when
pointing this at real cameras and a real controller, and `.env` is git-ignored.

**Licence.** MIT, see [`LICENSE`](LICENSE). No bundled fonts, images or audio;
all text rendering uses OpenCV's built-in Hershey vector fonts, and the
narration is synthesised offline at build time by the local system voice.

## Honest limitations

* Synthetic data only. Every number in [Results](#results) is a real
  measurement produced by `scripts/evaluate.py` on held-out seeds the model
  never trained on, and pasted into this file by `scripts/update_readme.py`
  rather than typed by hand — but held-out *synthetic* seeds. They say the
  architecture and the decision layer work; they say nothing about how the
  network transfers to real footage. Anyone deploying this would fine-tune on
  annotated real frames first.
* Kinematic arm, not dynamic. Grasp success is a positional tolerance, not a
  contact simulation.
* One arm. Throughput, not detection, is the main cause of hazards reaching the
  crusher in the end-to-end trials — the `too_late_flags` counter makes that
  visible rather than hiding it.
* The thermal model is physically motivated but not calibrated against real
  cells.
