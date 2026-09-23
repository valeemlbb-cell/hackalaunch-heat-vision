# Submission description

Paste-ready text for the HackaLaunch submission form. Replace the two
bracketed URLs after the repo is pushed and the video is uploaded.

---

**HeatVision — find the lithium in the waste stream and get it off the belt before the crusher does.**

Repo: <REPO_URL>
Demo (under 3 min): <VIDEO_URL>
Payout: 7W31iaCmjerN1jkpEnmZevn74SZxv83yEQvLsnc4PS7Q

A lithium cell in a shredder is a fire, and the reason this is still unsolved is
that neither obvious sensor works alone. A thermal camera false-alarms on every
hot brake disc in the stream; if heat alone stops the line, the line stops all
day and somebody switches the system off inside a week. A colour camera misses
the dangerous half: a discharged cell sits at ambient and looks exactly like a
black plastic shard.

HeatVision fuses a colour camera and a long-wave infrared camera at the first
convolution of one small network (HeatNet-S, 1.28 M parameters, CPU only,
~18 ms a frame), and then keeps the two signals separately accountable.
**Vision decides what an object is. Thermal decides how dangerous it is right
now.** Eight numbered rules combine them, and every decision is logged with the
rule id and the numbers that fired it — including the one that matters most: a
very hot object with no visual evidence of lithium raises an operator alert and
does **not** stop the line.

Then it acts. A 4-DOF arm solves an intercept for a target that is still
moving, flies a trajectory inside its acceleration limits, closes the gripper —
and checks whether it actually got it. A miss is recorded as a miss. A cell in
thermal runaway is never gripped at all: the belt halts and a human in PPE
lifts it into a quench drum, counted separately from the robot's own removals.

Measured on 400 held-out frames the model never trained on: **mAP@0.5 0.965,
precision 0.956, recall 0.973.** The numbers that actually matter are the
adversarial ones: **91.5% recall on discharged cells at ambient** (invisible to
thermal) and a **9.9% false-alarm rate on hot non-battery decoys** that would
trip a thermal-only trigger. End to end, in closed-loop runs on scenes it has
never seen, it keeps hazards out of the crusher and reports exactly how many it
missed, and why.

Simulation only — no physical hardware and no real lithium cells were used,
heated or damaged. The waste-stream simulator, the radiometric LWIR model, the
arm and the evaluation were all written for this hackathon from an empty
directory; nothing is pre-existing work. The dataset is 100% self-generated and
seed-addressed, so the held-out split regenerates byte-for-byte from a manifest.
Built with an AI coding agent under human direction; no AI service is called at
runtime and there are no API keys anywhere in the repo. MIT licensed.

The full autonomy disclosure — what is live, what is scripted (the narration
timeline, the cards, and an elevated hazard rate that is also shown on screen
throughout the video) — is in `docs/AUTONOMY.md`, and the safety case is in
`docs/SAFETY.md`.
