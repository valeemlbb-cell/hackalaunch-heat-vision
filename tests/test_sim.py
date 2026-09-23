"""Scene model and renderer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from heatvision.config import DEFAULT
from heatvision.sim.materials import BY_NAME, NEGATIVES, POSITIVES
from heatvision.sim.render import (
    ground_truth,
    item_box_px,
    px_to_m,
    m_to_px,
    render_frame,
    render_thermal,
    to_network_input,
)
from heatvision.sim.scene import Item, Scene, _instantiate


def _core_patch(thermal, cfg, item, half: int = 1):
    """3x3 patch at the item centroid: thin cells are only a few pixels wide."""
    from heatvision.sim.render import m_to_px

    col, row = m_to_px(cfg.belt, item.x_m, item.y_m)
    c, r = int(round(col)), int(round(row))
    return thermal[r - half : r + half + 1, c - half : c + half + 1]


def _single_item_scene(name: str, *, x=0.5, y=0.0, theta=0.0, seed=3) -> Scene:
    rng = np.random.default_rng(seed)
    scene = Scene(belt=DEFAULT.belt)
    item = _instantiate(rng, BY_NAME[name], 0, x, y)
    item.theta = theta
    scene.items.append(item)
    return scene


class TestScene:
    def test_populate_places_items_inside_the_belt(self, rng, cfg):
        scene = Scene.populate(rng, cfg.belt, n_items=(8, 12))
        assert 1 <= len(scene.items) <= 12
        limit = 0.5 * cfg.belt.width_m
        for item in scene.items:
            assert -limit <= item.y_m <= limit
            assert 0.0 <= item.x_m <= cfg.belt.view_len_m

    def test_populate_rejects_near_total_occlusion(self, rng, cfg):
        scene = Scene.populate(rng, cfg.belt, n_items=(14, 14), max_overlap=0.2)
        for a, b in ((a, b) for i, a in enumerate(scene.items) for b in scene.items[i + 1 :]):
            gap = math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
            assert gap >= (a.radius_m + b.radius_m) * 0.79

    def test_advance_moves_items_downstream_and_crushes_them(self, cfg):
        scene = _single_item_scene("cell_18650_warm", x=1.10)
        scene.advance(0.5, speed_mps=0.35)
        assert scene.items[0].x_m == pytest.approx(1.275, abs=1e-6)
        assert scene.items[0].crushed is True

    def test_advance_heats_a_venting_cell(self):
        scene = _single_item_scene("pouch_swollen_venting")
        start = scene.items[0].dt_core
        scene.items[0].rise_rate = 4.0
        scene.advance(2.0)
        assert scene.items[0].dt_core == pytest.approx(start + 8.0, abs=1e-6)

    def test_removed_items_do_not_move_and_are_not_visible(self):
        scene = _single_item_scene("cell_18650_warm", x=0.4)
        scene.items[0].removed = True
        scene.advance(1.0)
        assert scene.items[0].x_m == pytest.approx(0.4)
        assert scene.visible() == []

    def test_corners_rotate_with_theta(self):
        item = _single_item_scene("device_phone", theta=0.0).items[0]
        flat = item.corners_m()
        item.theta = math.pi / 2
        turned = item.corners_m()
        assert (flat[:, 0].max() - flat[:, 0].min()) > (turned[:, 0].max() - turned[:, 0].min())


class TestGeometry:
    def test_pixel_metre_roundtrip(self, cfg):
        for x, y in ((0.0, 0.0), (0.7, -0.3), (1.19, 0.41)):
            col, row = m_to_px(cfg.belt, x, y)
            bx, by = px_to_m(cfg.belt, col, row)
            assert bx == pytest.approx(x, abs=1e-9)
            assert by == pytest.approx(y, abs=1e-9)

    def test_box_covers_the_item_footprint(self, cfg):
        scene = _single_item_scene("device_phone", x=0.6, theta=0.6)
        item = scene.items[0]
        x0, y0, x1, y1 = item_box_px(cfg.belt, item)
        width_m = (x1 - x0) / cfg.belt.px_per_m
        assert width_m >= min(item.long_m, item.short_m) * 0.9
        assert width_m <= item.long_m * 1.45


class TestRender:
    def test_frame_shapes_and_types(self, rng, scene, cfg):
        rgb, thermal = render_frame(rng, scene, cfg.thermal)
        assert rgb.shape == (cfg.belt.image_px, cfg.belt.image_px, 3)
        assert rgb.dtype == np.uint8
        assert thermal.shape == (cfg.belt.image_px, cfg.belt.image_px)
        assert thermal.dtype == np.float32

    def test_rendering_is_deterministic_for_a_seed(self, scene, cfg):
        a = render_frame(np.random.default_rng(99), scene, cfg.thermal)
        b = render_frame(np.random.default_rng(99), scene, cfg.thermal)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_a_venting_pouch_is_much_hotter_than_the_belt(self, cfg):
        scene = _single_item_scene("pouch_swollen_venting")
        thermal = render_thermal(np.random.default_rng(1), scene, cfg.thermal)
        x0, y0, x1, y1 = (int(v) for v in item_box_px(cfg.belt, scene.items[0]))
        inside = thermal[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2]
        assert float(np.median(inside)) - float(np.median(thermal)) > 15.0

    def test_low_emissivity_metal_hides_its_heat(self, cfg):
        """A shiny surface reports far less than its true temperature.

        This is the physical reason a thermal-only trigger is not enough.
        """
        scene = _single_item_scene("cell_18650_warm")
        item = scene.items[0]
        item.dt_core = 30.0
        item.emissivity = 0.9
        matte = render_thermal(np.random.default_rng(2), scene, cfg.thermal)
        item.emissivity = 0.1
        shiny = render_thermal(np.random.default_rng(2), scene, cfg.thermal)
        hot = float(np.median(_core_patch(matte, cfg, item)))
        cold = float(np.median(_core_patch(shiny, cfg, item)))
        assert hot > cold + 15.0

    def test_cold_battery_is_invisible_to_thermal(self, cfg):
        scene = _single_item_scene("cell_discharged_cold")
        scene.items[0].dt_core = 0.4
        thermal = render_thermal(np.random.default_rng(4), scene, cfg.thermal)
        inside = _core_patch(thermal, cfg, scene.items[0])
        assert abs(float(np.median(inside)) - float(np.median(thermal))) < 3.0


class TestGroundTruth:
    def test_only_lithium_items_are_labelled(self, rng, cfg):
        scene = Scene.populate(rng, cfg.belt, n_items=(12, 12), hazard_rate=0.5)
        boxes, labels, uids = ground_truth(scene)
        by_uid = {it.uid: it for it in scene.items}
        assert len(boxes) == len(labels) == len(uids)
        for uid in uids:
            assert by_uid[uid].label is not None

    def test_boxes_stay_inside_the_frame(self, rng, cfg):
        scene = Scene.populate(rng, cfg.belt, n_items=(12, 12))
        boxes, _labels, _uids = ground_truth(scene)
        if len(boxes):
            assert boxes.min() >= 0.0
            assert boxes.max() <= cfg.belt.image_px - 1.0

    def test_network_input_is_four_channels_in_range(self, rng, scene, cfg):
        rgb, thermal = render_frame(rng, scene, cfg.thermal)
        tensor = to_network_input(rgb, thermal, cfg.detector, cfg.thermal)
        assert tensor.shape == (4, cfg.detector.input_px, cfg.detector.input_px)
        assert tensor[:3].min() >= 0.0 and tensor[:3].max() <= 1.0
        assert tensor[3].min() >= -0.2 and tensor[3].max() <= 1.2


class TestMaterials:
    def test_positives_carry_a_class_and_negatives_do_not(self):
        assert all(m.label is not None for m in POSITIVES)
        assert all(m.label is None for m in NEGATIVES)

    def test_the_adversarial_subsets_exist(self):
        tags = {m.subset for m in POSITIVES + NEGATIVES if m.subset}
        assert {"cold_battery", "thermal_event", "hot_decoy", "cell_lookalike"} <= tags

    def test_hot_decoys_outrank_ordinary_cells_on_temperature(self):
        decoy = BY_NAME["hot_motor_fragment"].dt_core[1]
        cell = BY_NAME["cell_18650_warm"].dt_core[1]
        assert decoy > cell, "the decoy must be able to beat a real cell on heat alone"
