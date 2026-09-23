"""CenterNet target encoding and the torch ``Dataset`` used for training."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import NUM_CLASSES, DetectorConfig
from ..sim.dataset import Sample


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """Radius at which a shifted box still has ``min_overlap`` IoU (CenterNet)."""
    a1 = 1.0
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))) / (2 * a1)

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (-b3 + math.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return max(1.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: float) -> None:
    """Splat a 2-D gaussian into ``heatmap`` in place, keeping the max."""
    r = int(max(1, round(radius)))
    sigma = (2 * r + 1) / 6.0
    ys, xs = np.ogrid[-r : r + 1, -r : r + 1]
    g = np.exp(-(xs**2 + ys**2) / (2 * sigma**2)).astype(np.float32)
    h, w = heatmap.shape
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    gx0, gy0 = x0 - (cx - r), y0 - (cy - r)
    patch = g[gy0 : gy0 + (y1 - y0), gx0 : gx0 + (x1 - x0)]
    np.maximum(heatmap[y0:y1, x0:x1], patch, out=heatmap[y0:y1, x0:x1])


def encode_targets(
    boxes: np.ndarray, labels: np.ndarray, cfg: DetectorConfig, num_classes: int = NUM_CLASSES
) -> dict[str, np.ndarray]:
    """Boxes in input pixels -> heatmap / size / offset / mask tensors."""
    out = cfg.out_px
    hm = np.zeros((num_classes, out, out), np.float32)
    wh = np.zeros((2, out, out), np.float32)
    off = np.zeros((2, out, out), np.float32)
    mask = np.zeros((1, out, out), np.float32)

    for (x0, y0, x1, y1), cls in zip(boxes, labels):
        bw = (x1 - x0) / cfg.stride
        bh = (y1 - y0) / cfg.stride
        if bw <= 0 or bh <= 0:
            continue
        cx = ((x0 + x1) * 0.5) / cfg.stride
        cy = ((y0 + y1) * 0.5) / cfg.stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < out and 0 <= iy < out):
            continue
        draw_gaussian(hm[int(cls)], ix, iy, gaussian_radius(bh, bw))
        wh[0, iy, ix] = bw
        wh[1, iy, ix] = bh
        off[0, iy, ix] = cx - ix
        off[1, iy, ix] = cy - iy
        mask[0, iy, ix] = 1.0
    return {"hm": hm, "wh": wh, "off": off, "mask": mask}


def _flip_boxes(boxes: np.ndarray, size: int, axis: str) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    out = boxes.copy()
    if axis == "x":
        out[:, 0], out[:, 2] = size - boxes[:, 2], size - boxes[:, 0]
    else:
        out[:, 1], out[:, 3] = size - boxes[:, 3], size - boxes[:, 1]
    return out


class HeatVisionDataset(Dataset):
    """Wraps pre-rendered samples and emits CenterNet targets.

    Augmentation is deliberately conservative: belt-plane flips (the sorter can
    run either way round) and a small thermal gain/offset jitter that models
    sensor drift and ambient change. No colour jitter beyond what the renderer
    already applies, and nothing that would break the RGB/LWIR registration.
    """

    def __init__(
        self,
        samples: list[Sample],
        cfg: DetectorConfig,
        *,
        augment: bool = False,
        seed: int = 0,
        zero_channels: tuple[int, ...] = (),
    ) -> None:
        self.samples = samples
        self.cfg = cfg
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        #: channels blanked before the network sees them, used for the
        #: single-modality ablations (RGB-only / thermal-only)
        self.zero_channels = zero_channels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = sample.image.copy()
        boxes = sample.boxes.copy()
        labels = sample.labels.copy()

        if self.augment:
            size = self.cfg.input_px
            if self.rng.random() < 0.5:
                image = image[:, :, ::-1].copy()
                boxes = _flip_boxes(boxes, size, "x")
            if self.rng.random() < 0.5:
                image = image[:, ::-1, :].copy()
                boxes = _flip_boxes(boxes, size, "y")
            gain = float(self.rng.normal(1.0, 0.06))
            bias = float(self.rng.normal(0.0, 0.012))
            image[3] = np.clip(image[3] * gain + bias, -0.2, 1.2)

        for ch in self.zero_channels:
            image[ch] = 0.0

        targets = encode_targets(boxes, labels, self.cfg)
        return {
            "image": torch.from_numpy(image),
            "hm": torch.from_numpy(targets["hm"]),
            "wh": torch.from_numpy(targets["wh"]),
            "off": torch.from_numpy(targets["off"]),
            "mask": torch.from_numpy(targets["mask"]),
        }
