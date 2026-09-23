"""Turn raw heatmaps into boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..config import CLASS_NAMES, DetectorConfig


@dataclass
class Detection:
    """One detected lithium item, in render-pixel coordinates."""

    box: tuple[float, float, float, float]
    score: float
    cls_id: int

    @property
    def cls_name(self) -> str:
        return CLASS_NAMES[self.cls_id]

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x1 - x0, y1 - y0)


def _peak_nms(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    return heat * (pooled == heat).float()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


@torch.no_grad()
def decode_batch(
    outputs: dict[str, torch.Tensor], cfg: DetectorConfig, scale: float = 1.0
) -> list[list[Detection]]:
    """Decode a batch of head outputs into per-image detection lists.

    ``scale`` maps network-input pixels back to whatever coordinate frame the
    caller wants (render pixels, typically ``render_px / input_px``).
    """
    heat = _peak_nms(torch.sigmoid(outputs["hm"]))
    batch, n_cls, out_h, out_w = heat.shape
    k = min(cfg.max_detections * 2, out_h * out_w)
    flat = heat.view(batch, -1)
    topk_scores, topk_idx = torch.topk(flat, k)

    cls_ids = torch.div(topk_idx, out_h * out_w, rounding_mode="floor")
    pix = topk_idx % (out_h * out_w)
    ys = torch.div(pix, out_w, rounding_mode="floor").float()
    xs = (pix % out_w).float()

    wh = outputs["wh"].view(batch, 2, -1)
    off = outputs["off"].view(batch, 2, -1)
    gather = pix.unsqueeze(1).expand(-1, 2, -1)
    wh_sel = torch.gather(wh, 2, gather)
    off_sel = torch.gather(off, 2, gather)

    cx = (xs + off_sel[:, 0]) * cfg.stride
    cy = (ys + off_sel[:, 1]) * cfg.stride
    bw = wh_sel[:, 0].clamp(min=0) * cfg.stride
    bh = wh_sel[:, 1].clamp(min=0) * cfg.stride

    x0 = (cx - bw / 2) * scale
    y0 = (cy - bh / 2) * scale
    x1 = (cx + bw / 2) * scale
    y1 = (cy + bh / 2) * scale

    results: list[list[Detection]] = []
    for b in range(batch):
        keep_mask = topk_scores[b] >= cfg.score_threshold
        boxes_np = (
            torch.stack([x0[b], y0[b], x1[b], y1[b]], dim=1)[keep_mask].cpu().numpy().astype(np.float32)
        )
        scores_np = topk_scores[b][keep_mask].cpu().numpy().astype(np.float32)
        cls_np = cls_ids[b][keep_mask].cpu().numpy().astype(np.int64)

        dets: list[Detection] = []
        for c in range(n_cls):
            sel = cls_np == c
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            for j in nms(boxes_np[idx], scores_np[idx], cfg.nms_iou):
                g = idx[j]
                dets.append(
                    Detection(
                        box=tuple(float(v) for v in boxes_np[g]),  # type: ignore[arg-type]
                        score=float(scores_np[g]),
                        cls_id=int(c),
                    )
                )
        dets.sort(key=lambda d: -d.score)
        results.append(dets[: cfg.max_detections])
    return results
