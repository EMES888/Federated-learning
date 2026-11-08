"""Tiny procedurally generated dataset (coloured shapes on textured background) for smoke tests /
CI: no download, deterministic. Each group has a different colour cast = feature-shift non-IID."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from .common import DataBundle, Sample

CLASSES = ["background", "circle", "rectangle", "triangle"]
PALETTE = np.array([[0, 0, 0], [220, 60, 60], [60, 200, 60], [60, 60, 220]], dtype=np.uint8)
TINTS = [(1.0, 1.0, 1.0), (1.2, 0.9, 0.8), (0.7, 0.8, 1.2), (0.6, 0.6, 0.6)]


def _make(path_img: str, path_mask: str, seed: int, group: int, size: int = 96) -> None:
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size, 3) * 60 + 80).astype(np.float32)
    mask = Image.new("L", (size, size), 0)
    canvas = Image.fromarray(img.astype(np.uint8))
    d_img, d_mask = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
    for _ in range(rng.randint(1, 4)):
        cls = rng.randint(1, 4)
        x0, y0 = rng.randint(0, size - 30, 2)
        w, h = rng.randint(15, 40, 2)
        box = [int(x0), int(y0), int(min(x0 + w, size - 1)), int(min(y0 + h, size - 1))]
        color = tuple(int(c) for c in PALETTE[cls] * 0.8 + rng.randint(0, 40, 3))
        if cls == 1:
            d_img.ellipse(box, fill=color); d_mask.ellipse(box, fill=cls)
        elif cls == 2:
            d_img.rectangle(box, fill=color); d_mask.rectangle(box, fill=cls)
        else:
            tri = [(box[0], box[3]), (box[2], box[3]), ((box[0] + box[2]) // 2, box[1])]
            d_img.polygon(tri, fill=color); d_mask.polygon(tri, fill=cls)
    arr = np.asarray(canvas).astype(np.float32) * np.array(TINTS[group % len(TINTS)])
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(path_img)
    mask.save(path_mask)


def load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint8)


def build(root: str, download: bool = True, n_train: int = 96, n_val: int = 16, n_test: int = 24,
          n_groups: int = 4) -> DataBundle:
    base = os.path.join(root, "synthetic")
    os.makedirs(base, exist_ok=True)

    def split(name, n, offset):
        out = []
        for i in range(n):
            g = i % n_groups
            pi, pm = os.path.join(base, f"{name}_{i}.png"), os.path.join(base, f"{name}_{i}_m.png")
            if not (os.path.exists(pi) and os.path.exists(pm)):
                _make(pi, pm, offset + i, g)
            out.append(Sample(pi, pm, g))
        return out

    return DataBundle(name="synthetic", train=split("train", n_train, 0),
                      val=split("val", n_val, 10_000), test=split("test", n_test, 20_000),
                      class_names=CLASSES, palette=PALETTE, load_mask=load_mask, img_size=(64, 64),
                      group_names={g: f"domain{g}" for g in range(n_groups)})
