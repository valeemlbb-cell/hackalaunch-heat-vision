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
