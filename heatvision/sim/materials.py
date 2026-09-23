"""Material catalogue for the waste stream.

The point of this file is the *hard negatives*. A thermal-only system trips on
a warm brake motor; an RGB-only system misses a matte-black discharged cell
lying next to a black plastic shard. Both failure modes are represented here on
purpose so the evaluation can measure them.

``dt_core`` is kelvin above ambient at the hottest point of the item.
``emissivity`` scales how much of that the LWIR camera actually sees; bare metal
(0.05-0.2) mirrors the room instead of reporting its own temperature.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    #: detector class, or ``None`` for background / hard negative
    label: str | None
    #: base BGR colour (OpenCV order), 0-255
    color: tuple[int, int, int]
    #: colour jitter magnitude per channel
    color_jitter: int
    #: footprint on the belt, metres (long side, short side) ranges
    size_long_m: tuple[float, float]
    size_short_m: tuple[float, float]
    #: kelvin above ambient at the core
    dt_core: tuple[float, float]
    #: LWIR emissivity
    emissivity: tuple[float, float]
    #: how fast the item is heating, K/s (only non-zero for thermal events)
    rise_rate: tuple[float, float] = (0.0, 0.0)
    #: drawn as an ellipse rather than a rotated rectangle
    round_shape: bool = False
    #: rough/specular surface -> speckle in the RGB channel
    specular: float = 0.0
    #: relative sampling weight when populating a scene
    weight: float = 1.0
    #: marks a deliberately adversarial sample, tracked in the eval report
    subset: str | None = None


# --------------------------------------------------------------------------
# Lithium-bearing items (positives)
# --------------------------------------------------------------------------
POSITIVES: tuple[Material, ...] = (
    Material(
        name="cell_18650_warm",
        label="cell",
        color=(48, 52, 58),
        color_jitter=18,
        size_long_m=(0.060, 0.072),
        size_short_m=(0.017, 0.021),
        dt_core=(5.0, 13.0),
        emissivity=(0.45, 0.70),  # shrink-wrapped, so not bare metal
        specular=0.25,
        weight=1.6,
    ),
    Material(
        name="cell_aa_wrapped",
        label="cell",
        color=(92, 96, 104),
        color_jitter=22,
        size_long_m=(0.044, 0.053),
        size_short_m=(0.013, 0.016),
        dt_core=(3.0, 8.0),
        emissivity=(0.50, 0.75),
        specular=0.20,
        weight=1.0,
    ),
    Material(
        name="cell_discharged_cold",
        label="cell",
        color=(34, 36, 40),
        color_jitter=10,
        size_long_m=(0.058, 0.070),
        size_short_m=(0.017, 0.020),
        dt_core=(0.0, 1.2),  # indistinguishable from the belt in LWIR
        emissivity=(0.45, 0.70),
        specular=0.30,
        weight=1.0,
        subset="cold_battery",
    ),
    Material(
        name="pouch_lipo",
        label="pouch",
        color=(140, 92, 40),
        color_jitter=26,
        size_long_m=(0.055, 0.095),
        size_short_m=(0.032, 0.058),
        dt_core=(7.0, 20.0),
        emissivity=(0.80, 0.92),
        weight=1.2,
    ),
    Material(
        name="pouch_swollen_venting",
        label="pouch",
        color=(120, 104, 96),
        color_jitter=30,
        size_long_m=(0.060, 0.105),
        size_short_m=(0.040, 0.070),
        dt_core=(34.0, 72.0),  # thermal event in progress
        emissivity=(0.82, 0.94),
        rise_rate=(2.5, 7.0),
        weight=0.8,
        subset="thermal_event",
    ),
    Material(
        name="device_phone",
        label="device",
        color=(58, 60, 66),
        color_jitter=20,
        size_long_m=(0.120, 0.155),
        size_short_m=(0.062, 0.078),
        dt_core=(2.5, 9.0),
        emissivity=(0.60, 0.88),
        specular=0.35,
        weight=1.1,
    ),
    Material(
        name="device_vape",
        label="device",
        color=(76, 132, 152),
        color_jitter=34,
        size_long_m=(0.085, 0.115),
        size_short_m=(0.018, 0.026),
        dt_core=(3.0, 11.0),
        emissivity=(0.55, 0.85),
        weight=1.2,
    ),
    Material(
        name="device_powerbank",
        label="device",
        color=(180, 176, 168),
        color_jitter=24,
        size_long_m=(0.095, 0.135),
        size_short_m=(0.048, 0.066),
        dt_core=(4.0, 14.0),
        emissivity=(0.35, 0.65),
        specular=0.30,
        weight=0.9,
    ),
)

# --------------------------------------------------------------------------
# Background clutter and adversarial negatives
# --------------------------------------------------------------------------
NEGATIVES: tuple[Material, ...] = (
    Material(
        name="cardboard",
        label=None,
        color=(96, 132, 168),
        color_jitter=26,
        size_long_m=(0.080, 0.210),
        size_short_m=(0.060, 0.150),
        dt_core=(0.0, 1.0),
        emissivity=(0.88, 0.95),
        weight=2.2,
    ),
    Material(
        name="pet_bottle",
        label=None,
        color=(190, 205, 195),
        color_jitter=22,
        size_long_m=(0.150, 0.230),
        size_short_m=(0.055, 0.080),
        dt_core=(0.0, 0.8),
        emissivity=(0.85, 0.94),
        round_shape=True,
        specular=0.45,
        weight=1.8,
    ),
    Material(
        name="alu_can",
        label=None,
        color=(198, 200, 204),
        color_jitter=18,
        size_long_m=(0.100, 0.125),
        size_short_m=(0.058, 0.068),
        dt_core=(0.0, 1.5),
        emissivity=(0.06, 0.16),  # mirrors the room, reads cold
        round_shape=True,
        specular=0.65,
        weight=1.6,
    ),
    Material(
        name="pcb_scrap",
        label=None,
        color=(42, 96, 44),
        color_jitter=22,
        size_long_m=(0.055, 0.130),
        size_short_m=(0.038, 0.090),
        dt_core=(0.0, 2.0),
        emissivity=(0.75, 0.90),
        specular=0.20,
        weight=1.4,
    ),
    Material(
        name="fabric",
        label=None,
        color=(120, 78, 130),
        color_jitter=44,
        size_long_m=(0.090, 0.200),
        size_short_m=(0.070, 0.160),
        dt_core=(0.0, 1.2),
        emissivity=(0.90, 0.96),
        weight=1.3,
    ),
    Material(
        name="black_plastic_shard",
        label=None,
        color=(30, 31, 34),
        color_jitter=8,
        size_long_m=(0.045, 0.085),
        size_short_m=(0.016, 0.034),
        dt_core=(0.0, 1.0),
        emissivity=(0.88, 0.95),
        weight=1.7,
        subset="cell_lookalike",
    ),
    Material(
        name="steel_bolt_shiny",
        label=None,
        color=(168, 172, 178),
        color_jitter=16,
        size_long_m=(0.040, 0.075),
        size_short_m=(0.012, 0.020),
        dt_core=(0.0, 1.0),
        emissivity=(0.10, 0.25),
        specular=0.70,
        weight=1.2,
        subset="cell_lookalike",
    ),
    Material(
        name="hot_motor_fragment",
        label=None,
        color=(112, 116, 120),
        color_jitter=20,
        size_long_m=(0.070, 0.120),
        size_short_m=(0.055, 0.095),
        dt_core=(24.0, 58.0),  # hotter than most real batteries
        emissivity=(0.70, 0.88),
        rise_rate=(0.0, 1.2),
        weight=1.0,
        subset="hot_decoy",
    ),
    Material(
        name="hot_brake_disc",
        label=None,
        color=(130, 134, 138),
        color_jitter=18,
        size_long_m=(0.110, 0.165),
        size_short_m=(0.100, 0.150),
        dt_core=(18.0, 40.0),
        emissivity=(0.55, 0.80),
        round_shape=True,
        weight=0.7,
        subset="hot_decoy",
    ),
    Material(
        name="foil_wrap",
        label=None,
        color=(206, 208, 210),
        color_jitter=14,
        size_long_m=(0.050, 0.110),
        size_short_m=(0.030, 0.070),
        dt_core=(0.0, 1.0),
        emissivity=(0.04, 0.12),
        specular=0.80,
        weight=1.1,
    ),
)

ALL_MATERIALS: tuple[Material, ...] = POSITIVES + NEGATIVES
BY_NAME = {m.name: m for m in ALL_MATERIALS}

#: adversarial subsets reported separately in the evaluation
SUBSETS: tuple[str, ...] = (
    "cold_battery",
    "thermal_event",
    "hot_decoy",
    "cell_lookalike",
)
