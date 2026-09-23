# Architecture

```
                 ┌──────────────┐   ┌──────────────┐
  belt  ────────▶│ colour cam   │   │ LWIR cam     │   pre-registered, same grid
                 └──────┬───────┘   └──────┬───────┘
                        │  R,G,B           │  dT (K above ambient)
                        └────────┬─────────┘
                                 ▼
                      ┌──────────────────────┐
                      │ HeatNet-S            │  4-ch early fusion, stride-4
                      │ 1.28 M params, CPU   │  CenterNet head
                      └──────────┬───────────┘
                                 │ boxes + class + score
                                 ▼
          ┌──────────────────────────────────────────┐
          │ thermal.measure()  — peak dT inside box  │ independent of the net
          └──────────────────────┬───────────────────┘
                                 ▼
                      ┌──────────────────────┐
                      │ BeltTracker          │  persistence + dT/dt
                      └──────────┬───────────┘
                                 ▼
                      ┌──────────────────────┐
                      │ policy.decide()      │  R0..R7, explicit + logged
                      └──────────┬───────────┘
             ┌───────────────────┼────────────────────┐
             ▼                   ▼                    ▼
     EMERGENCY_STOP       EXTRACT_QUENCH        EXTRACT_QUARANTINE
     belt halt,           belt stopped,         pick on the fly,
     human callout        gentle 5 N grip       12 N grip
             │                   └─────────┬──────────┘
             │                             ▼
             │                   ┌──────────────────────┐
             │                   │ Arm                  │ intercept, trajectory,
             │                   │ 4 DOF + gripper      │ grasp verification
             │                   └──────────┬───────────┘
             ▼                              ▼
     operator extraction            quarantine / quench bin
```

## Module map

| Path | Responsibility |
| --- | --- |
| `heatvision/config.py` | every physical constant and threshold, one place |
| `heatvision/sim/materials.py` | the waste-stream catalogue, including the adversarial negatives |
| `heatvision/sim/scene.py` | item placement, belt motion, self-heating |
| `heatvision/sim/render.py` | RGB rasteriser and the separate LWIR radiometric model |
| `heatvision/sim/dataset.py` | seed-addressed splits, manifest, digests |
| `heatvision/detect/model.py` | HeatNet-S, focal + L1 losses |
| `heatvision/detect/targets.py` | CenterNet target encoding, augmentation, ablation masks |
| `heatvision/detect/decode.py` | peak NMS, box decode, IoU / NMS helpers |
| `heatvision/detect/train.py` | CPU training loop with cosine schedule |
| `heatvision/detect/infer.py` | checkpoint loading, frame-pair inference |
| `heatvision/decide/thermal.py` | belt baseline, in-box dT, rise-rate fit |
| `heatvision/decide/tracker.py` | belt-aware association, persistence, thermal history |
| `heatvision/decide/policy.py` | the rule set, hazard tiers, grip-force selection |
| `heatvision/arm/kinematics.py` | FK / IK / workspace / grasp angle |
| `heatvision/arm/trajectory.py` | trapezoidal profiles with synchronised arrival |
| `heatvision/arm/robot.py` | pick state machine, intercept solve, grasp verification |
| `heatvision/pipeline.py` | the closed loop and its statistics |
| `heatvision/viz/` | operator console and demo cards |
| `heatvision/evaluation/` | AP / mAP, subset analysis, end-to-end trials |

## Why early fusion, and why only 1.28 M parameters

Late fusion (two backbones, merge the boxes) cannot express "dark cylinder that
is 8 K warmer than the belt" — by the time the two streams meet, the joint cue
is gone. Early fusion learns it in the first convolution, at the price of
needing registered frames, which a bispectral module gives you anyway.

The size is a deployment constraint, not a limitation we are apologising for. A
sort line runs an edge box in a dusty cabinet, not an A100. HeatNet-S trains in
about twenty minutes on a laptop CPU and infers in tens of milliseconds on the
same CPU, so the whole loop closes without a GPU anywhere on the plant floor.

## Coordinate frames

Everything is in **belt metres**. `x` runs downstream, `y` runs across the belt,
the origin is the upstream edge of the camera window at the belt centre line.
The camera is isotropic, so `render.m_to_px` / `px_to_m` is a single scale plus
an offset, and the arm's kinematics are in the same frame — there is no
hidden calibration step between "the detector said here" and "the arm went
there". On real hardware that mapping is exactly the homography that
`HEATVISION_REGISTRATION` points at.

## Control timing

The station runs at 20 Hz. The arm integrates trajectories continuously, so it
is not tied to the perception rate; the interface between them is
`Arm.step(dt)`, which advances whatever segment is active and returns a
`PickResult` when the pick concludes. Segment boundaries can overshoot by up to
one control period, which is accounted for in the intercept budget.
