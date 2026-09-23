"""Detection metrics and the dataset's split hygiene."""

from __future__ import annotations

import numpy as np
import pytest

from heatvision.config import DEFAULT
from heatvision.detect.decode import Detection
from heatvision.evaluation.metrics import (
    ImageGroundTruth,
    average_precision,
    evaluate_detections,
    subset_report,
)
from heatvision.sim.dataset import (
    SPLITS,
    build_split,
    render_sample,
    split_digest,
    split_seeds,
    write_manifest,
)


def det(box, score=0.9, cls_id=0) -> Detection:
    return Detection(box=tuple(float(v) for v in box), score=score, cls_id=cls_id)


def gt(boxes, labels, **kwargs) -> ImageGroundTruth:
    return ImageGroundTruth(
        boxes=np.array(boxes, np.float32).reshape(-1, 4),
        labels=np.array(labels, np.int64),
        **kwargs,
    )


class TestAveragePrecision:
    def test_perfect_curve_scores_one(self):
        recall = np.array([0.5, 1.0])
        precision = np.array([1.0, 1.0])
        assert average_precision(recall, precision) == pytest.approx(1.0)

    def test_empty_curve_scores_zero(self):
        assert average_precision(np.array([]), np.array([])) == 0.0

    def test_half_recall_at_full_precision(self):
        recall = np.array([0.5, 0.5])
        precision = np.array([1.0, 0.5])
        assert average_precision(recall, precision) == pytest.approx(0.5)


class TestEvaluateDetections:
    def test_perfect_predictions_score_one(self):
        boxes = [[10, 10, 30, 30]]
        report = evaluate_detections([[det(boxes[0])]], [gt(boxes, [0])])
        assert report["mAP@0.5"] == pytest.approx(1 / 3, abs=1e-3)  # two classes have no GT
        cls = report["by_iou"]["0.50"]["per_class"]["cell"]
        assert cls["ap"] == pytest.approx(1.0)
        assert cls["precision"] == pytest.approx(1.0)
        assert cls["recall"] == pytest.approx(1.0)

    def test_a_wrong_class_is_a_miss_and_a_false_positive(self):
        boxes = [[10, 10, 30, 30]]
        report = evaluate_detections([[det(boxes[0], cls_id=1)]], [gt(boxes, [0])])
        per_class = report["by_iou"]["0.50"]["per_class"]
        assert per_class["cell"]["recall"] == 0.0
        assert per_class["pouch"]["fp"] == 1

    def test_duplicate_detections_count_once(self):
        boxes = [[10, 10, 30, 30]]
        preds = [[det(boxes[0], 0.9), det([11, 11, 31, 31], 0.8)]]
        report = evaluate_detections(preds, [gt(boxes, [0])])
        cls = report["by_iou"]["0.50"]["per_class"]["cell"]
        assert cls["tp"] == 1
        assert cls["fp"] == 1

    def test_a_poorly_localised_box_fails_the_iou_gate(self):
        preds = [[det([10, 10, 60, 60])]]
        report = evaluate_detections(preds, [gt([[10, 10, 30, 30]], [0])])
        assert report["by_iou"]["0.50"]["per_class"]["cell"]["tp"] == 0

    def test_low_scoring_predictions_are_excluded_from_the_operating_point(self):
        boxes = [[10, 10, 30, 30]]
        report = evaluate_detections(
            [[det(boxes[0], score=0.1)]], [gt(boxes, [0])], operating_threshold=0.3
        )
        cls = report["by_iou"]["0.50"]["per_class"]["cell"]
        assert cls["tp"] == 0
        assert cls["ap"] == pytest.approx(1.0)  # AP is threshold free

    def test_class_agnostic_view_merges_the_classes(self):
        preds = [[det([10, 10, 30, 30], cls_id=2)]]
        report = evaluate_detections(preds, [gt([[10, 10, 30, 30]], [2])])
        agn = report["by_iou"]["0.50"]["class_agnostic"]
        assert agn["tp"] == 1 and agn["fp"] == 0 and agn["fn"] == 0
        assert agn["f1"] == pytest.approx(1.0)

    def test_multiple_iou_thresholds_give_a_range_metric(self):
        boxes = [[10, 10, 30, 30]]
        report = evaluate_detections(
            [[det(boxes[0])]], [gt(boxes, [0])], iou_thresholds=(0.5, 0.75, 0.95)
        )
        assert report["mAP@[0.5:0.95]"] is not None
        assert set(report["by_iou"]) == {"0.50", "0.75", "0.95"}

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            evaluate_detections([[]], [gt([[0, 0, 1, 1]], [0]), gt([[0, 0, 1, 1]], [0])])


class TestSubsetReport:
    def test_a_hit_on_a_tagged_positive_counts_as_recall(self):
        boxes = [[10, 10, 30, 30]]
        report = subset_report([[det(boxes[0])]], [gt(boxes, [0], subsets=("cold_battery",))])
        assert report["cold_battery"]["recall"] == pytest.approx(1.0)
        assert report["cold_battery"]["n"] == 1

    def test_a_box_on_a_hot_decoy_counts_as_a_false_alarm(self):
        ground = gt(
            [[10, 10, 30, 30]],
            [0],
            neg_boxes=np.array([[60, 60, 90, 90]], np.float32),
            neg_subsets=("hot_decoy",),
        )
        report = subset_report([[det([60, 60, 90, 90])]], [ground])
        assert report["NEG:hot_decoy"]["false_alarm_rate"] == pytest.approx(1.0)

    def test_a_clean_run_has_no_false_alarms(self):
        ground = gt(
            [[10, 10, 30, 30]],
            [0],
            neg_boxes=np.array([[60, 60, 90, 90]], np.float32),
            neg_subsets=("hot_decoy",),
        )
        report = subset_report([[det([10, 10, 30, 30])]], [ground])
        assert report["NEG:hot_decoy"]["false_alarm_rate"] == 0.0


class TestSplitHygiene:
    def test_the_splits_do_not_share_a_single_seed(self):
        seeds = {name: set(split_seeds(name, 4000)) for name in SPLITS}
        assert seeds["train"].isdisjoint(seeds["val"])
        assert seeds["train"].isdisjoint(seeds["test"])
        assert seeds["val"].isdisjoint(seeds["test"])

    def test_an_unknown_split_is_rejected(self):
        with pytest.raises(ValueError):
            split_seeds("holdout", 10)

    def test_a_seed_always_renders_the_same_frame(self):
        a = render_sample(9_000_000)
        b = render_sample(9_000_000)
        assert np.array_equal(a.image, b.image)
        assert np.array_equal(a.boxes, b.boxes)
        assert np.array_equal(a.labels, b.labels)

    def test_different_seeds_render_different_frames(self):
        a = render_sample(9_000_000)
        b = render_sample(9_000_001)
        assert not np.array_equal(a.image, b.image)

    def test_the_test_split_digest_is_reproducible(self):
        first = build_split("test", 8, DEFAULT)
        second = build_split("test", 8, DEFAULT)
        assert split_digest(first) == split_digest(second)

    def test_samples_carry_subset_tags_and_negative_boxes(self):
        found_positive_tag = False
        found_negative_tag = False
        for seed in range(9_000_000, 9_000_030):
            sample = render_sample(seed)
            found_positive_tag |= any(t for t in sample.subsets)
            found_negative_tag |= any(t for t in sample.neg_subsets)
        assert found_positive_tag and found_negative_tag

    def test_manifest_records_the_seed_ranges(self, tmp_path):
        manifest = write_manifest(tmp_path / "manifest.json", {"train": 10, "val": 5, "test": 5})
        assert (tmp_path / "manifest.json").exists()
        assert manifest["splits"]["test"]["count"] == 5
        assert manifest["splits"]["train"]["seed_base"] != manifest["splits"]["test"]["seed_base"]
