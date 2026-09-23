from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heatvision.config import CLASS_INDEX, DEFAULT  # noqa: E402
from heatvision.detect.decode import Detection  # noqa: E402
from heatvision.detect.model import HeatNetS  # noqa: E402
from heatvision.sim.render import item_box_px  # noqa: E402
from heatvision.sim.scene import Scene  # noqa: E402


@pytest.fixture
def cfg():
    return DEFAULT


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def scene(rng, cfg):
    return Scene.populate(rng, cfg.belt, n_items=(6, 10))


@pytest.fixture
def untrained_model(cfg):
    return HeatNetS(cfg.detector)


class OracleDetector:
    """Perfect detector wired directly to the simulator's ground truth.

    Used to test the *rest* of the loop — tracking, policy, arm, scoring —
    without the trained network's noise. It is a test double only; nothing in
    ``heatvision/`` imports it.
    """

    def __init__(self, cfg=DEFAULT, score: float = 0.92, jitter_px: float = 0.0, seed: int = 0):
        self.cfg = cfg
        self.zero_channels: tuple[int, ...] = ()
        self.meta: dict = {"oracle": True}
        self.score = score
        self.jitter_px = jitter_px
        self.station = None
        self._rng = np.random.default_rng(seed)

    def detect(self, rgb, thermal):  # noqa: ANN001 - test double
        if self.station is None:
            return []
        out: list[Detection] = []
        for item in self.station.scene.visible():
            if item.label is None:
                continue
            x0, y0, x1, y1 = item_box_px(self.cfg.belt, item)
            if self.jitter_px:
                dx, dy = self._rng.normal(0, self.jitter_px, 2)
                x0, x1 = x0 + dx, x1 + dx
                y0, y1 = y0 + dy, y1 + dy
            out.append(Detection(box=(x0, y0, x1, y1), score=self.score, cls_id=CLASS_INDEX[item.label]))
        return out


@pytest.fixture
def oracle_detector():
    return OracleDetector()
