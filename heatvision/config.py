"""Central configuration for the HeatVision cell.

Every magic number in this project lives here so that the simulator, the
detector, the decision policy and the arm controller all agree on units.

Units
-----
length: metres, angle: radians, temperature: kelvin above ambient (dT) unless a
name ends in ``_c`` (degrees Celsius), time: seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
# Only lithium-bearing items are detector classes. Everything else is
# background, including deliberately hard negatives (hot motor fragments,
# battery-shaped foil, shiny cans).
CLASS_NAMES: tuple[str, ...] = (
    "cell",  # cylindrical 18650 / AA style lithium cell
    "pouch",  # LiPo pouch, possibly swollen / damaged
    "device",  # e-waste device with an embedded cell (phone, vape, power bank)
)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# --------------------------------------------------------------------------
# Camera / belt geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BeltConfig:
    """Physical layout of the inspection station."""

    width_m: float = 0.90  # cross-belt (image y)
    view_len_m: float = 1.20  # along-belt visible window (image x)
    speed_mps: float = 0.35  # nominal belt speed
    ambient_c: float = 22.0
    surface_dt: float = 0.6  # belt rubber runs slightly warm
    # Camera looks straight down; both sensors are pre-registered.
    image_px: int = 256  # rendered RGB/thermal frame is square
    crusher_x_m: float = 1.15  # past this x the item is unrecoverable

    @property
    def px_per_m(self) -> float:
        return self.image_px / self.view_len_m

    @property
    def belt_half_px(self) -> float:
        """Half the belt width in pixels (the camera is isotropic)."""
        return 0.5 * self.width_m * self.px_per_m

    @property
    def center_row(self) -> float:
        return 0.5 * self.image_px


# --------------------------------------------------------------------------
# Thermal sensor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalConfig:
    """Uncooled LWIR microbolometer model (FLIR Lepton class)."""

    noise_equivalent_dt: float = 0.055  # NETD in kelvin
    fixed_pattern_sigma: float = 0.18  # per-sensor column noise
    span_k: float = 60.0  # dT mapped into 0..1 for the network
    blur_sigma_px: float = 1.1  # optics + lower native resolution
    reflected_ambient_c: float = 24.0  # what low-emissivity metal mirrors


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorConfig:
    input_px: int = 128
    stride: int = 4  # output heatmap is input_px / stride
    width: int = 32  # base channel count
    in_channels: int = 4  # R, G, B, thermal
    score_threshold: float = 0.30
    nms_iou: float = 0.45
    max_detections: int = 20

    @property
    def out_px(self) -> int:
        return self.input_px // self.stride


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch_size: int = 16
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    warmup_steps: int = 60
    hm_weight: float = 1.0
    wh_weight: float = 0.10
    off_weight: float = 1.0
    seed: int = 1337
    num_workers: int = 0  # CPU training, keep it deterministic


# --------------------------------------------------------------------------
# Decision policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds for the hazard tiers.

    The policy never trusts a single modality. ``vision_*`` gates come from the
    detector, ``dt_*`` gates come from the calibrated thermal frame.
    """

    vision_accept: float = 0.55  # confident lithium item -> extract
    vision_review: float = 0.30  # weak hit, needs thermal corroboration
    dt_warm: float = 6.0  # K above belt baseline: self-heating cell
    dt_hot: float = 22.0  # K: thermal event in progress
    dt_runaway: float = 45.0  # K: venting / runaway, do not touch
    rise_rate_runaway: float = 3.5  # K/s sustained rise -> runaway
    min_track_frames: int = 2  # frames an item must persist before we act
    max_reach_x_m: float = 1.02  # beyond this the arm cannot make the pick
    grip_force_limit_n: float = 12.0  # puncture risk above this
    swollen_force_limit_n: float = 5.0  # damaged pouches get a gentle grip


# --------------------------------------------------------------------------
# Arm
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmConfig:
    """Planar SCARA-style 3R arm + vertical Z axis + gripper (4 DOF + grip)."""

    base_xy_m: tuple[float, float] = (0.55, -0.34)  # off to the side of the belt
    link1_m: float = 0.42
    link2_m: float = 0.38
    z_travel_m: float = 0.30
    z_pick_m: float = 0.015  # tool height when gripping something on the belt
    z_clear_m: float = 0.18  # safe transit height
    # Limits taken from the published envelope of a mid-size industrial SCARA
    # (standard 25/300/25 mm cycle in roughly 0.35 s), not from wishful thinking.
    joint_vel_limit: tuple[float, float, float] = (7.5, 8.5, 12.0)  # rad/s
    joint_acc_limit: tuple[float, float, float] = (35.0, 45.0, 60.0)  # rad/s^2
    z_vel_limit: float = 1.8  # m/s
    z_acc_limit: float = 18.0  # m/s^2
    grip_time_s: float = 0.18
    release_time_s: float = 0.12
    dt: float = 1.0 / 60.0  # control period
    # Drop targets (x, y) in belt frame.
    quarantine_bin_m: tuple[float, float] = (0.30, -0.62)
    quench_bin_m: tuple[float, float] = (0.78, -0.62)  # sand-filled, for hot items


@dataclass(frozen=True)
class CellConfig:
    """The whole station."""

    belt: BeltConfig = field(default_factory=BeltConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)


DEFAULT = CellConfig()
