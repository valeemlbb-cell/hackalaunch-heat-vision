"""Detection metrics: precision, recall, AP, mAP.

Implemented from scratch (no pycocotools dependency) using the standard
greedy, score-ordered, one-to-one matching:

* sort every prediction of a class across the whole split by score,
* match each to the highest-IoU unmatched ground truth in its own image,
* below the IoU threshold, or if the ground truth is already taken, it is a
  false positive,
* AP is the area under the resulting precision/recall curve with the
  monotone-decreasing precision envelope (VOC 2010 / COCO "all-points" rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CLASS_NAMES, NUM_CLASSES
from ..detect.decode import Detection, iou_matrix


@dataclass
class ImageGroundTruth:
    boxes: np.ndarray  # (N, 4)
    labels: np.ndarray  # (N,)
    subsets: tuple[str, ...] = ()
    neg_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    neg_subsets: tuple[str, ...] = ()


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision/recall curve, all-points interpolation."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):  # monotone envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match_class(
    predictions: list[tuple[int, Detection]],
    ground_truth: list[ImageGroundTruth],
    cls_id: int,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, list[tuple[int, int, int]]]:
    """Greedy matching for one class.

    Returns ``(tp, fp, n_gt, matches)`` where ``matches`` lists
    ``(image_index, prediction_rank, gt_index)`` for each true positive.
    """
    gt_boxes = [
        g.boxes[g.labels == cls_id] if len(g.labels) else np.zeros((0, 4), np.float32)
        for g in ground_truth
    ]
    gt_map = [np.nonzero(g.labels == cls_id)[0] if len(g.labels) else np.zeros(0, int) for g in ground_truth]
    n_gt = int(sum(len(b) for b in gt_boxes))
    taken = [np.zeros(len(b), dtype=bool) for b in gt_boxes]

    preds = sorted(predictions, key=lambda p: -p[1].score)
    tp = np.zeros(len(preds), np.float32)
    fp = np.zeros(len(preds), np.float32)
    matches: list[tuple[int, int, int]] = []

    for rank, (img_idx, det) in enumerate(preds):
        boxes = gt_boxes[img_idx]
        if len(boxes) == 0:
            fp[rank] = 1.0
            continue
        ious = iou_matrix(np.array([det.box], np.float32), boxes)[0]
        ious = np.where(taken[img_idx], -1.0, ious)
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            taken[img_idx][best] = True
            tp[rank] = 1.0
            matches.append((img_idx, rank, int(gt_map[img_idx][best])))
        else:
            fp[rank] = 1.0
    return tp, fp, n_gt, matches


def evaluate_detections(
    predictions: list[list[Detection]],
    ground_truth: list[ImageGroundTruth],
    *,
    iou_thresholds: tuple[float, ...] = (0.5,),
    operating_threshold: float = 0.30,
) -> dict:
    """Full detection report over a split."""
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground truth must be the same length")

    per_iou: dict[str, dict] = {}
    for iou_t in iou_thresholds:
        per_class: dict[str, dict] = {}
        for cls_id in range(NUM_CLASSES):
            flat = [
                (img_idx, det)
                for img_idx, dets in enumerate(predictions)
                for det in dets
                if det.cls_id == cls_id
            ]
            tp, fp, n_gt, _ = _match_class(flat, ground_truth, cls_id, iou_t)
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
            recall = cum_tp / max(n_gt, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
            ap = average_precision(recall, precision)

            scores = np.array([d.score for _, d in sorted(flat, key=lambda p: -p[1].score)], np.float32)
            keep = scores >= operating_threshold
            n_keep = int(keep.sum())
            op_tp = float(tp[:n_keep].sum())
            op_fp = float(fp[:n_keep].sum())
            per_class[CLASS_NAMES[cls_id]] = {
                "ap": round(ap, 4),
                "n_gt": n_gt,
                "n_pred": len(flat),
                "precision": round(op_tp / max(op_tp + op_fp, 1e-9), 4),
                "recall": round(op_tp / max(n_gt, 1), 4),
                "tp": int(op_tp),
                "fp": int(op_fp),
                "fn": int(n_gt - op_tp),
            }

        # class-agnostic "is it lithium at all" numbers, which is what the
        # safety case actually cares about
        agnostic = _agnostic(predictions, ground_truth, iou_t, operating_threshold)
        per_iou[f"{iou_t:.2f}"] = {
            "per_class": per_class,
            "mAP": round(float(np.mean([v["ap"] for v in per_class.values()])), 4),
            "class_agnostic": agnostic,
        }

    primary = per_iou[f"{iou_thresholds[0]:.2f}"]
    return {
        "operating_threshold": operating_threshold,
        "mAP@0.5": per_iou.get("0.50", primary)["mAP"],
        "mAP@[0.5:0.95]": round(float(np.mean([v["mAP"] for v in per_iou.values()])), 4)
        if len(per_iou) > 1
        else None,
        "by_iou": per_iou,
    }


def _agnostic(
    predictions: list[list[Detection]],
    ground_truth: list[ImageGroundTruth],
    iou_t: float,
    operating_threshold: float,
) -> dict:
    """Treat all three classes as one "lithium hazard" class."""
    flat = [
        (img_idx, det)
        for img_idx, dets in enumerate(predictions)
        for det in dets
        if det.score >= operating_threshold
    ]
    gt_boxes = [g.boxes for g in ground_truth]
    n_gt = int(sum(len(b) for b in gt_boxes))
    taken = [np.zeros(len(b), dtype=bool) for b in gt_boxes]
    tp = fp = 0
    for img_idx, det in sorted(flat, key=lambda p: -p[1].score):
        boxes = gt_boxes[img_idx]
        if len(boxes) == 0:
            fp += 1
            continue
        ious = iou_matrix(np.array([det.box], np.float32), boxes)[0]
        ious = np.where(taken[img_idx], -1.0, ious)
        best = int(np.argmax(ious))
        if ious[best] >= iou_t:
            taken[img_idx][best] = True
            tp += 1
        else:
            fp += 1
    return {
        "precision": round(tp / max(tp + fp, 1e-9), 4),
        "recall": round(tp / max(n_gt, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + (n_gt - tp), 1e-9), 4),
        "tp": tp,
        "fp": fp,
        "fn": n_gt - tp,
        "n_gt": n_gt,
    }


def subset_report(
    predictions: list[list[Detection]],
    ground_truth: list[ImageGroundTruth],
    *,
    iou_threshold: float = 0.5,
    operating_threshold: float = 0.30,
) -> dict:
    """Recall on adversarial positives and false-alarm rate on decoys.

    This is the table that says whether the fusion actually bought anything:
    ``cold_battery`` recall is what RGB has to carry on its own, and the
    ``hot_decoy`` false-alarm rate is what a thermal-only trigger would get
    catastrophically wrong.
    """
    pos: dict[str, list[int]] = {}
    for img_idx, gt in enumerate(ground_truth):
        dets = [d for d in predictions[img_idx] if d.score >= operating_threshold]
        det_boxes = np.array([d.box for d in dets], np.float32) if dets else np.zeros((0, 4), np.float32)
        if len(gt.boxes):
            ious = iou_matrix(gt.boxes, det_boxes) if len(det_boxes) else np.zeros((len(gt.boxes), 0))
            hit = ious.max(axis=1) >= iou_threshold if ious.size else np.zeros(len(gt.boxes), bool)
            for k, tag in enumerate(gt.subsets or ("",) * len(gt.boxes)):
                key = tag or "ordinary"
                pos.setdefault(key, [0, 0])
                pos[key][1] += 1
                pos[key][0] += int(hit[k])
        if len(gt.neg_boxes):
            ious = (
                iou_matrix(gt.neg_boxes, det_boxes) if len(det_boxes) else np.zeros((len(gt.neg_boxes), 0))
            )
            alarm = ious.max(axis=1) >= 0.3 if ious.size else np.zeros(len(gt.neg_boxes), bool)
            for k, tag in enumerate(gt.neg_subsets or ("",) * len(gt.neg_boxes)):
                key = f"NEG:{tag or 'ordinary'}"
                pos.setdefault(key, [0, 0])
                pos[key][1] += 1
                pos[key][0] += int(alarm[k])

    out: dict[str, dict] = {}
    for key, (hits, total) in sorted(pos.items()):
        rate = hits / total if total else 0.0
        out[key] = {
            "n": total,
            ("false_alarm_rate" if key.startswith("NEG:") else "recall"): round(rate, 4),
            "hits": hits,
        }
    return out
