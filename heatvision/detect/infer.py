"""Inference wrapper: frame pair in, detections out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import DEFAULT, CellConfig
from ..sim.render import to_network_input
from .decode import Detection, decode_batch
from .model import HeatNetS

CHECKPOINT_NAME = "heatnet_s.pt"


def save_checkpoint(path: Path, model: HeatNetS, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


class Detector:
    """Loads a trained HeatNet-S and runs it on registered RGB + LWIR frames."""

    def __init__(
        self,
        model: HeatNetS,
        cfg: CellConfig = DEFAULT,
        device: str = "cpu",
        zero_channels: tuple[int, ...] = (),
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        #: must match whatever the checkpoint was trained with
        self.zero_channels = zero_channels
        self.meta: dict = {}

    @classmethod
    def load(cls, path: Path, cfg: CellConfig = DEFAULT, device: str = "cpu") -> "Detector":
        blob = torch.load(path, map_location=device, weights_only=False)
        model = HeatNetS(cfg.detector)
        model.load_state_dict(blob["state_dict"])
        meta = blob.get("meta", {})
        det = cls(model, cfg, device, zero_channels=tuple(meta.get("zero_channels", ())))
        det.meta = meta
        return det

    @property
    def render_scale(self) -> float:
        return self.cfg.belt.image_px / self.cfg.detector.input_px

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, thermal: np.ndarray) -> list[Detection]:
        tensor = to_network_input(rgb, thermal, self.cfg.detector, self.cfg.thermal)
        for ch in self.zero_channels:
            tensor[ch] = 0.0
        batch = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        outputs = self.model(batch)
        return decode_batch(outputs, self.cfg.detector, scale=self.render_scale)[0]

    @torch.no_grad()
    def detect_tensor_batch(self, images: torch.Tensor, scale: float | None = None) -> list[list[Detection]]:
        outputs = self.model(images.to(self.device))
        return decode_batch(
            outputs, self.cfg.detector, scale=self.render_scale if scale is None else scale
        )
