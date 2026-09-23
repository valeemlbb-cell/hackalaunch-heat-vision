"""HeatNet-S: a small anchor-free (CenterNet-style) bispectral detector.

Four input channels (R, G, B, calibrated LWIR) go into one encoder. There is no
separate thermal branch: early fusion lets the first convolutions learn joint
features such as "dark cylinder that is 8 K warmer than the belt", which is
exactly the cue that separates a lithium cell from a black plastic shard.

Roughly 1.1 M parameters, trains on a laptop CPU in minutes, runs at ~30 FPS on
the same CPU — the intended deployment target is an edge box bolted to a sort
line, not a datacentre GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES, DetectorConfig


def conv_bn(in_ch: int, out_ch: int, stride: int = 1, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.a = conv_bn(ch, ch)
        self.b = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.b(self.a(x)))


class HeatNetS(nn.Module):
    """Encoder / decoder with a stride-4 detection head."""

    def __init__(self, cfg: DetectorConfig | None = None, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        cfg = cfg or DetectorConfig()
        self.cfg = cfg
        w = cfg.width

        self.stem = nn.Sequential(conv_bn(cfg.in_channels, w, stride=2), ResBlock(w))  # /2
        self.down1 = nn.Sequential(conv_bn(w, 2 * w, stride=2), ResBlock(2 * w))  # /4
        self.down2 = nn.Sequential(conv_bn(2 * w, 4 * w, stride=2), ResBlock(4 * w))  # /8
        self.down3 = nn.Sequential(conv_bn(4 * w, 4 * w, stride=2), ResBlock(4 * w))  # /16

        self.lat2 = nn.Conv2d(4 * w, 4 * w, 1)
        self.lat1 = nn.Conv2d(2 * w, 4 * w, 1)
        self.fuse2 = conv_bn(4 * w, 4 * w)
        self.fuse1 = conv_bn(4 * w, 2 * w)

        self.head_stem = nn.Sequential(conv_bn(2 * w, 2 * w), ResBlock(2 * w))
        self.hm = nn.Conv2d(2 * w, num_classes, 1)
        self.wh = nn.Conv2d(2 * w, 2, 1)
        self.off = nn.Conv2d(2 * w, 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # focal-loss friendly prior: start by predicting "mostly background"
        nn.init.constant_(self.hm.bias, -4.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s2 = self.stem(x)
        s4 = self.down1(s2)
        s8 = self.down2(s4)
        s16 = self.down3(s8)

        up = F.interpolate(s16, size=s8.shape[-2:], mode="nearest")
        p8 = self.fuse2(up + self.lat2(s8))
        up = F.interpolate(p8, size=s4.shape[-2:], mode="nearest")
        p4 = self.fuse1(up + self.lat1(s4))

        feat = self.head_stem(p4)
        return {
            "hm": self.hm(feat),  # logits
            "wh": self.wh(feat),  # box size in output cells
            "off": self.off(feat),  # sub-cell centre offset
        }

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CornerNet / CenterNet penalty-reduced focal loss."""
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1.0 - 1e-4)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(pred) * (1.0 - pred).pow(alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * pred.pow(alpha) * (1.0 - target).pow(beta) * neg
    n_pos = pos.sum()
    if n_pos < 1:
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 over the positive cells only. ``mask`` is ``(B, 1, H, W)``."""
    denom = mask.sum() * pred.shape[1] + 1e-4
    return (torch.abs(pred - target) * mask).sum() / denom
