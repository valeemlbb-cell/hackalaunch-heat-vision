"""Detector: target encoding, model wiring, decoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from heatvision.config import NUM_CLASSES, DetectorConfig
from heatvision.detect.decode import decode_batch, iou_matrix, nms
from heatvision.detect.model import HeatNetS, focal_loss, masked_l1
from heatvision.detect.targets import (
    HeatVisionDataset,
    draw_gaussian,
    encode_targets,
    gaussian_radius,
)
from heatvision.sim.dataset import render_sample


class TestTargets:
    def test_gaussian_radius_grows_with_box_size(self):
        assert gaussian_radius(10, 10) < gaussian_radius(40, 40)
        assert gaussian_radius(4, 4) >= 1.0

    def test_draw_gaussian_peaks_at_the_centre(self):
        hm = np.zeros((16, 16), np.float32)
        draw_gaussian(hm, 8, 8, 3.0)
        assert hm[8, 8] == pytest.approx(1.0, abs=1e-5)
        assert hm[8, 8] > hm[8, 10] > hm[8, 13]

    def test_draw_gaussian_clips_at_the_border(self):
        hm = np.zeros((16, 16), np.float32)
        draw_gaussian(hm, 0, 0, 4.0)
        assert hm[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert np.isfinite(hm).all()

    def test_encode_marks_exactly_one_positive_cell_per_box(self):
        cfg = DetectorConfig()
        boxes = np.array([[20.0, 20.0, 44.0, 36.0], [80.0, 90.0, 96.0, 104.0]], np.float32)
        labels = np.array([0, 2], np.int64)
        t = encode_targets(boxes, labels, cfg)
        assert t["mask"].sum() == 2
        assert t["hm"].shape == (NUM_CLASSES, cfg.out_px, cfg.out_px)
        assert t["hm"][0].max() == pytest.approx(1.0, abs=1e-5)
        assert t["hm"][1].max() == 0.0

    def test_encode_stores_size_and_offset_in_output_cells(self):
        cfg = DetectorConfig()
        boxes = np.array([[20.0, 20.0, 44.0, 36.0]], np.float32)
        t = encode_targets(boxes, np.array([0], np.int64), cfg)
        iy, ix = np.nonzero(t["mask"][0])
        assert t["wh"][0, iy[0], ix[0]] == pytest.approx(24.0 / cfg.stride)
        assert t["wh"][1, iy[0], ix[0]] == pytest.approx(16.0 / cfg.stride)
        assert 0.0 <= t["off"][0, iy[0], ix[0]] < 1.0

    def test_encode_ignores_boxes_outside_the_frame(self):
        cfg = DetectorConfig()
        boxes = np.array([[500.0, 500.0, 520.0, 520.0]], np.float32)
        t = encode_targets(boxes, np.array([1], np.int64), cfg)
        assert t["mask"].sum() == 0


class TestDataset:
    def test_dataset_emits_all_target_tensors(self):
        cfg = DetectorConfig()
        samples = [render_sample(1_000_000 + i) for i in range(2)]
        ds = HeatVisionDataset(samples, cfg)
        item = ds[0]
        assert item["image"].shape == (4, cfg.input_px, cfg.input_px)
        assert item["hm"].shape == (NUM_CLASSES, cfg.out_px, cfg.out_px)
        assert item["mask"].shape == (1, cfg.out_px, cfg.out_px)

    def test_ablation_blanks_the_requested_channels(self):
        cfg = DetectorConfig()
        samples = [render_sample(1_000_000)]
        ds = HeatVisionDataset(samples, cfg, zero_channels=(3,))
        assert float(ds[0]["image"][3].abs().sum()) == 0.0
        assert float(ds[0]["image"][0].abs().sum()) > 0.0

    def test_flip_augmentation_keeps_the_same_number_of_targets(self):
        cfg = DetectorConfig()
        sample = render_sample(1_000_004)
        plain = HeatVisionDataset([sample], cfg, augment=False)[0]
        flipped = HeatVisionDataset([sample], cfg, augment=True, seed=1)[0]
        assert float(plain["mask"].sum()) == float(flipped["mask"].sum())


class TestModel:
    def test_forward_shapes(self, cfg):
        model = HeatNetS(cfg.detector)
        out = model(torch.zeros(2, 4, cfg.detector.input_px, cfg.detector.input_px))
        assert out["hm"].shape == (2, NUM_CLASSES, cfg.detector.out_px, cfg.detector.out_px)
        assert out["wh"].shape[1] == 2
        assert out["off"].shape[1] == 2

    def test_model_is_small_enough_for_an_edge_box(self, cfg):
        assert HeatNetS(cfg.detector).num_parameters() < 3_000_000

    def test_heatmap_bias_starts_near_background(self, cfg):
        model = HeatNetS(cfg.detector)
        prior = torch.sigmoid(model.hm.bias).mean().item()
        assert prior < 0.05

    def test_focal_loss_rewards_a_correct_peak(self):
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 4, 4] = 1.0
        good = torch.full((1, 1, 8, 8), -6.0)
        good[0, 0, 4, 4] = 6.0
        bad = torch.full((1, 1, 8, 8), -6.0)
        bad[0, 0, 1, 1] = 6.0
        assert float(focal_loss(good, target)) < float(focal_loss(bad, target))

    def test_masked_l1_only_counts_positive_cells(self):
        pred = torch.ones(1, 2, 4, 4)
        target = torch.zeros(1, 2, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        assert float(masked_l1(pred, target, mask)) == pytest.approx(0.0)
        mask[0, 0, 2, 2] = 1.0
        assert float(masked_l1(pred, target, mask)) == pytest.approx(1.0, abs=1e-3)

    def test_gradients_flow_to_every_head(self, cfg):
        model = HeatNetS(cfg.detector)
        out = model(torch.randn(1, 4, cfg.detector.input_px, cfg.detector.input_px))
        (out["hm"].sum() + out["wh"].sum() + out["off"].sum()).backward()
        assert model.stem[0][0].weight.grad is not None
        assert model.hm.weight.grad is not None


class TestDecode:
    def _synthetic_outputs(self, cfg: DetectorConfig, cx: int, cy: int, bw: float, bh: float, cls: int):
        hm = torch.full((1, NUM_CLASSES, cfg.out_px, cfg.out_px), -10.0)
        hm[0, cls, cy, cx] = 10.0
        wh = torch.zeros(1, 2, cfg.out_px, cfg.out_px)
        wh[0, 0, cy, cx] = bw
        wh[0, 1, cy, cx] = bh
        off = torch.zeros(1, 2, cfg.out_px, cfg.out_px)
        return {"hm": hm, "wh": wh, "off": off}

    def test_decode_recovers_a_planted_box(self, cfg):
        det_cfg = cfg.detector
        out = self._synthetic_outputs(det_cfg, cx=8, cy=10, bw=6.0, bh=4.0, cls=1)
        dets = decode_batch(out, det_cfg, scale=1.0)[0]
        assert len(dets) == 1
        d = dets[0]
        assert d.cls_id == 1
        assert d.score > 0.99
        assert d.center == pytest.approx((8 * det_cfg.stride, 10 * det_cfg.stride), abs=1e-4)
        assert d.size == pytest.approx((6.0 * det_cfg.stride, 4.0 * det_cfg.stride), abs=1e-4)

    def test_scale_maps_back_to_render_pixels(self, cfg):
        out = self._synthetic_outputs(cfg.detector, 8, 8, 5.0, 5.0, 0)
        small = decode_batch(out, cfg.detector, scale=1.0)[0][0]
        big = decode_batch(out, cfg.detector, scale=2.0)[0][0]
        assert big.center[0] == pytest.approx(small.center[0] * 2)

    def test_low_scores_are_dropped(self, cfg):
        out = self._synthetic_outputs(cfg.detector, 8, 8, 5.0, 5.0, 0)
        out["hm"][0, 0, 8, 8] = -1.5  # sigmoid ~0.18, below the 0.30 threshold
        assert decode_batch(out, cfg.detector)[0] == []

    def test_detections_come_back_sorted_by_score(self, cfg):
        out = self._synthetic_outputs(cfg.detector, 6, 6, 4.0, 4.0, 0)
        out["hm"][0, 0, 20, 20] = 2.0
        out["wh"][0, :, 20, 20] = 4.0
        dets = decode_batch(out, cfg.detector)[0]
        assert len(dets) == 2
        assert dets[0].score >= dets[1].score


class TestBoxOps:
    def test_iou_of_identical_boxes_is_one(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[20, 20, 30, 30]], np.float32)
        assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_iou_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[5, 0, 15, 10]], np.float32)
        assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150, abs=1e-5)

    def test_nms_keeps_the_best_of_an_overlapping_pair(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [40, 40, 50, 50]], np.float32)
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        assert nms(boxes, scores, 0.45) == [0, 2]

    def test_nms_on_empty_input(self):
        assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5) == []
