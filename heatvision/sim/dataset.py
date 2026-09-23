"""Dataset construction.

The dataset is *procedural and seed-addressed*: a split is a list of integer
seeds, and seed ``k`` always renders to exactly the same frame. That gives a
genuinely held-out test set (the seeds never touch training) that anybody can
regenerate byte-for-byte from ``data/manifest.json`` without a multi-gigabyte
download. ``tests/test_determinism.py`` pins the test-split hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DEFAULT, BeltConfig, CellConfig
from .render import ground_truth, item_box_px, render_frame, to_network_input
from .scene import Scene

SPLITS = ("train", "val", "test")

#: disjoint seed bases keep the splits from ever colliding
_SPLIT_BASE = {"train": 1_000_000, "val": 5_000_000, "test": 9_000_000}


@dataclass
class Sample:
    image: np.ndarray  # (4, H, W) float32 network input
    boxes: np.ndarray  # (N, 4) in *network input* pixels
    labels: np.ndarray  # (N,) int64
    seed: int
    #: adversarial subset tag per ground-truth box ("" when ordinary)
    subsets: tuple[str, ...] = ()
    #: boxes of *negative* objects, used for hard-negative analysis only
    neg_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    neg_subsets: tuple[str, ...] = ()


def scene_for_seed(seed: int, belt: BeltConfig, *, hazard_rate: float = 0.42) -> tuple[Scene, np.random.Generator]:
    rng = np.random.default_rng(seed)
    scene = Scene.populate(rng, belt, hazard_rate=hazard_rate)
    return scene, rng


def render_sample(seed: int, cfg: CellConfig = DEFAULT, *, hazard_rate: float = 0.42) -> Sample:
    """Deterministically render one training sample from its seed."""
    scene, rng = scene_for_seed(seed, cfg.belt, hazard_rate=hazard_rate)
    rgb, thermal = render_frame(rng, scene, cfg.thermal)
    boxes, labels, uids = ground_truth(scene)
    scale = cfg.detector.input_px / cfg.belt.image_px
    image = to_network_input(rgb, thermal, cfg.detector, cfg.thermal)

    by_uid = {it.uid: it for it in scene.items}
    subsets = tuple((by_uid[u].subset or "") for u in uids)
    negatives = [it for it in scene.visible() if it.label is None]
    if negatives:
        neg_boxes = np.array([item_box_px(cfg.belt, it) for it in negatives], np.float32) * scale
        neg_subsets = tuple((it.subset or "") for it in negatives)
    else:
        neg_boxes = np.zeros((0, 4), np.float32)
        neg_subsets = ()

    return Sample(
        image=image,
        boxes=boxes * scale,
        labels=labels,
        seed=seed,
        subsets=subsets,
        neg_boxes=neg_boxes,
        neg_subsets=neg_subsets,
    )


def split_seeds(split: str, count: int) -> list[int]:
    if split not in _SPLIT_BASE:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    base = _SPLIT_BASE[split]
    return [base + i for i in range(count)]


def write_manifest(path: Path, counts: dict[str, int], cfg: CellConfig = DEFAULT) -> dict:
    """Persist the split definition so the dataset is reproducible."""
    manifest = {
        "generator": "heatvision.sim",
        "version": 1,
        "note": (
            "Fully synthetic. Seeds are the dataset: render_sample(seed) is "
            "deterministic, so these three seed ranges define disjoint splits."
        ),
        "belt": {
            "image_px": cfg.belt.image_px,
            "view_len_m": cfg.belt.view_len_m,
            "width_m": cfg.belt.width_m,
        },
        "detector_input_px": cfg.detector.input_px,
        "splits": {
            split: {
                "count": counts[split],
                "seed_base": _SPLIT_BASE[split],
                "seeds": [_SPLIT_BASE[split], _SPLIT_BASE[split] + counts[split] - 1],
            }
            for split in SPLITS
            if split in counts
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_split(
    split: str,
    count: int,
    cfg: CellConfig = DEFAULT,
    *,
    progress: bool = False,
) -> list[Sample]:
    samples: list[Sample] = []
    seeds = split_seeds(split, count)
    for i, seed in enumerate(seeds):
        samples.append(render_sample(seed, cfg))
        if progress and (i + 1) % 100 == 0:
            print(f"  {split}: {i + 1}/{count}", flush=True)
    return samples


def split_digest(samples: list[Sample]) -> str:
    """Stable SHA-256 over a split, used to prove the test set never moved."""
    h = hashlib.sha256()
    for s in samples:
        h.update(np.round(s.image * 255).astype(np.uint8).tobytes())
        h.update(np.round(s.boxes, 3).astype(np.float32).tobytes())
        h.update(s.labels.astype(np.int64).tobytes())
    return h.hexdigest()


def cache_path(root: Path, split: str, count: int, cfg: CellConfig = DEFAULT) -> Path:
    return Path(root) / f"{split}_{count}_{cfg.detector.input_px}.npz"


def save_split(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.stack([s.image for s in samples]).astype(np.float16),
        boxes=np.concatenate([s.boxes for s in samples]) if samples else np.zeros((0, 4), np.float32),
        labels=np.concatenate([s.labels for s in samples]) if samples else np.zeros((0,), np.int64),
        counts=np.array([len(s.labels) for s in samples], np.int64),
        seeds=np.array([s.seed for s in samples], np.int64),
    )


def load_split(path: Path) -> list[Sample]:
    data = np.load(path)
    images = data["images"].astype(np.float32)
    counts = data["counts"]
    boxes = data["boxes"]
    labels = data["labels"]
    seeds = data["seeds"]
    out: list[Sample] = []
    cursor = 0
    for i, n in enumerate(counts):
        n = int(n)
        out.append(
            Sample(
                image=images[i],
                boxes=boxes[cursor : cursor + n].astype(np.float32),
                labels=labels[cursor : cursor + n].astype(np.int64),
                seed=int(seeds[i]),
            )
        )
        cursor += n
    return out


def get_split(
    split: str,
    count: int,
    cfg: CellConfig = DEFAULT,
    *,
    cache_dir: Path | None = None,
    progress: bool = False,
) -> list[Sample]:
    """Render a split, using an on-disk cache when one is available."""
    if cache_dir is not None:
        path = cache_path(cache_dir, split, count, cfg)
        if path.exists():
            return load_split(path)
        samples = build_split(split, count, cfg, progress=progress)
        save_split(path, samples)
        return samples
    return build_split(split, count, cfg, progress=progress)
