# Safety case

## What this system is allowed to do

| Situation | Action | Rule |
| --- | --- | --- |
| Hot object, no visual evidence of lithium | operator alert, **line keeps running** | R0 |
| Cold clutter | ignore | R1 |
| Lithium item, venting or rising >3.5 K/s while already above 22 K | **belt halt**, suppression, human callout, **no grip** | R2 |
| Seen for fewer than 2 frames | observe | R3 |
| Past the 1.02 m pick window or out of the arm's workspace | flag the downstream diverter, do not chase | R4 |
| Lithium item above 22 K but stable | belt stopped, 5 N gentle grip, quench (sand) bin | R5 |
| Lithium item confirmed, cool | pick on the fly, 12 N grip, quarantine bin | R6 |
| Weak visual hit with no thermal support | observe | R7 |

Every decision carries the rule id and the numbers that fired it, and
`Station.log` records all of them with timestamps.

## The three properties the rules exist to guarantee

**1. Thermal never acts alone.** A brake disc off a scrap car can sit 40 K above
the belt. If heat alone could stop the line, the line would stop all day and
somebody would switch the system off inside a week — which is how these systems
actually fail. R0 requires visual evidence before heat means anything, and its
consequence is an alert, not a halt. `tests/test_decide.py::test_R0_hot_decoy_alerts_instead_of_stopping_the_line`
pins this behaviour, and the evaluation reports the hot-decoy false-alarm rate
as a first-class number.

**2. Vision never acts alone either.** A discharged cell is at ambient
temperature and looks like any other dark cylinder. R6 accepts a confident
visual hit on its own, but R5 and R2 escalate purely on measured temperature, so
an item that warms up between frames is re-graded on the spot rather than being
locked into the class it got on frame one.

**3. A failing cell is never gripped.** Above 45 K, or rising faster than
3.5 K/s once already hot, the correct response is not a robot pick — squeezing a
cell in thermal runaway spreads it. The arm aborts any job in flight, the belt
stops, and a human in PPE lifts the item into a quench drum. That manual
extraction is counted separately (`hazards_operator_removed`) from the robot's
own removals so the two are never conflated in the headline number.

## Grip force

Jaw force is chosen per item, not globally: 12 N for an intact cell or device,
5 N for a pouch or anything already warm. Puncturing a pouch is a direct
ignition path, so the gentle limit is applied to the whole `pouch` class
regardless of its temperature.

The gripper closes across the item's **short** axis (`kinematics.grasp_angle`),
never along its length and never onto the terminals of a cylindrical cell.

## Interlocks in the simulated cell

* Transit happens at `z_clear = 0.18 m`, above anything on the belt; the descent
  to `z_pick = 0.015 m` only starts once the tool is over the target.
* The arm refuses a job it cannot intercept inside its workspace
  (`Arm.start` returns `False`) instead of reaching and colliding.
* Only one job runs at a time; an emergency stop aborts whatever is in flight.
* A quench pick stops the belt before the arm enters the stream; the belt only
  restarts once the arm is idle again.

## What this repository does NOT do

* It does **not** test on, damage, puncture, short or heat any real lithium
  cell. There is no physical hardware in this project at all.
* It does not claim the synthetic thermal model is a substitute for calibration
  against real cells. It is a physically-motivated model
  (`apparent = e·T_object + (1-e)·T_reflected`, plus NETD and fixed-pattern
  noise) built so that the *decision logic* can be exercised and the failure
  modes reproduced, not a radiometric ground truth.
* It does not present anything scripted as autonomous. See
  [`docs/AUTONOMY.md`](AUTONOMY.md).

## Before this could run on a real line

1. Radiometric calibration of the LWIR camera against reference cells, and a
   real RGB↔LWIR homography (`HEATVISION_REGISTRATION`).
2. Re-training or fine-tuning on annotated real footage; the synthetic-only
   numbers in this README do not transfer unexamined.
3. A hardware safety chain that does not depend on this software: a physical
   E-stop, an interlocked guard, and fire suppression over the belt. The policy
   here is a supervisory layer, never the last line of defence.
4. Force-limited or compliant gripper hardware, with force actually measured
   rather than commanded.
