"""Waste-stream scene model.

A :class:`Scene` is a bag of :class:`Item` objects lying on a moving belt. The
same scene object is used for three different jobs:

* generating still training frames (``populate`` + render),
* driving the closed-loop demo (``advance`` moves and heats the items),
* scoring the pipeline (items carry ground truth and a ``removed`` flag).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..config import BeltConfig
from .materials import ALL_MATERIALS, NEGATIVES, POSITIVES, Material


@dataclass
class Item:
    """One object on the belt."""

    uid: int
    material: Material
    x_m: float
    y_m: float
    theta: float
    long_m: float
    short_m: float
    dt_core: float
    emissivity: float
    rise_rate: float
    color: tuple[int, int, int]
    #: set by the pipeline once the arm has taken it off the belt
    removed: bool = False
    #: set when the item has passed the crusher inlet
    crushed: bool = False

    @property
    def label(self) -> str | None:
        return self.material.label

    @property
    def is_hazard(self) -> bool:
        return self.material.label is not None

    @property
    def subset(self) -> str | None:
        return self.material.subset

    @property
    def radius_m(self) -> float:
        return 0.5 * math.hypot(self.long_m, self.short_m)

    def corners_m(self) -> np.ndarray:
        """Four corners of the oriented footprint, shape (4, 2)."""
        hl, hs = 0.5 * self.long_m, 0.5 * self.short_m
        base = np.array([[-hl, -hs], [hl, -hs], [hl, hs], [-hl, hs]], dtype=np.float64)
        c, s = math.cos(self.theta), math.sin(self.theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return base @ rot.T + np.array([self.x_m, self.y_m])

    def aabb_m(self) -> tuple[float, float, float, float]:
        """Axis-aligned box ``(x0, y0, x1, y1)`` in belt metres."""
        pts = self.corners_m()
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )


def _sample_material(rng: np.random.Generator, pool: tuple[Material, ...]) -> Material:
    weights = np.array([m.weight for m in pool], dtype=np.float64)
    weights /= weights.sum()
    return pool[int(rng.choice(len(pool), p=weights))]


def _instantiate(rng: np.random.Generator, material: Material, uid: int, x: float, y: float) -> Item:
    jitter = material.color_jitter
    color = tuple(
        int(np.clip(c + rng.integers(-jitter, jitter + 1), 0, 255)) for c in material.color
    )
    return Item(
        uid=uid,
        material=material,
        x_m=x,
        y_m=y,
        theta=float(rng.uniform(-math.pi, math.pi)),
        long_m=float(rng.uniform(*material.size_long_m)),
        short_m=float(rng.uniform(*material.size_short_m)),
        dt_core=float(rng.uniform(*material.dt_core)),
        emissivity=float(rng.uniform(*material.emissivity)),
        rise_rate=float(rng.uniform(*material.rise_rate)),
        color=color,  # type: ignore[arg-type]
    )


@dataclass
class Scene:
    belt: BeltConfig
    items: list[Item] = field(default_factory=list)
    time_s: float = 0.0
    _next_uid: int = 0

    # ---------------------------------------------------------------- build
    @classmethod
    def populate(
        cls,
        rng: np.random.Generator,
        belt: BeltConfig,
        *,
        n_items: tuple[int, int] = (5, 14),
        hazard_rate: float = 0.42,
        x_range: tuple[float, float] | None = None,
        max_place_tries: int = 40,
        max_overlap: float = 0.22,
    ) -> "Scene":
        """Drop a cluttered pile of junk on the belt.

        Items may touch and partially occlude one another (``max_overlap``) —
        a clean, well-separated scene would not be a realistic waste stream —
        but we reject near-total occlusion so the ground truth stays honest.
        """
        scene = cls(belt=belt)
        target = int(rng.integers(n_items[0], n_items[1] + 1))
        lo, hi = x_range if x_range is not None else (0.04, belt.view_len_m - 0.04)
        y_lim = 0.5 * belt.width_m - 0.035
        for _ in range(target):
            pool = POSITIVES if rng.random() < hazard_rate else NEGATIVES
            material = _sample_material(rng, pool)
            for _try in range(max_place_tries):
                x = float(rng.uniform(lo, hi))
                y = float(rng.uniform(-y_lim, y_lim))
                cand = _instantiate(rng, material, scene._next_uid, x, y)
                if scene._fits(cand, max_overlap):
                    scene.items.append(cand)
                    scene._next_uid += 1
                    break
        return scene

    def _fits(self, cand: Item, max_overlap: float) -> bool:
        for other in self.items:
            gap = math.hypot(cand.x_m - other.x_m, cand.y_m - other.y_m)
            touch = cand.radius_m + other.radius_m
            if gap < touch * (1.0 - max_overlap):
                return False
        return True

    def spawn(self, rng: np.random.Generator, *, hazard_rate: float = 0.42) -> Item:
        """Feed one new item in at the upstream edge of the view."""
        pool = POSITIVES if rng.random() < hazard_rate else NEGATIVES
        material = _sample_material(rng, pool)
        y_lim = 0.5 * self.belt.width_m - 0.035
        item = _instantiate(
            rng,
            material,
            self._next_uid,
            x=-0.05,
            y=float(rng.uniform(-y_lim, y_lim)),
        )
        self._next_uid += 1
        self.items.append(item)
        return item

    # ---------------------------------------------------------------- step
    def advance(self, dt: float, *, speed_mps: float | None = None) -> None:
        """Move the belt on by ``dt`` seconds and let hot items keep heating."""
        speed = self.belt.speed_mps if speed_mps is None else speed_mps
        self.time_s += dt
        for item in self.items:
            if item.removed:
                continue
            item.x_m += speed * dt
            if item.rise_rate > 0.0:
                item.dt_core += item.rise_rate * dt
            if item.x_m > self.belt.crusher_x_m:
                item.crushed = True

    def visible(self) -> list[Item]:
        """Items currently inside the camera window and still on the belt."""
        return [
            it
            for it in self.items
            if not it.removed and not it.crushed and -0.02 <= it.x_m <= self.belt.view_len_m + 0.02
        ]

    def hazards_remaining(self) -> list[Item]:
        return [it for it in self.items if it.is_hazard and not it.removed and not it.crushed]

    def shifted(self, dx_m: float) -> "Scene":
        """Copy of the scene translated along the belt (used for augmentation)."""
        return Scene(
            belt=self.belt,
            items=[replace(it, x_m=it.x_m + dx_m) for it in self.items],
            time_s=self.time_s,
            _next_uid=self._next_uid,
        )


def material_names() -> list[str]:
    return [m.name for m in ALL_MATERIALS]
