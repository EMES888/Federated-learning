"""Dataset-agnostic pieces: sample container, paired image/mask transforms, label statistics."""
from __future__ import annotations

import os
import random
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Sample:
    image: str                 # path to RGB image
    mask: str                  # path to label image
    group: int = 0             # "natural" client id: camera sequence / city / breed ...


@dataclass
class DataBundle:
    name: str
    train: List[Sample]
    val: List[Sample]
    test: List[Sample]
    class_names: List[str]
    palette: np.ndarray                       # (C, 3) uint8 for visualisation
    load_mask: Callable[[str], np.ndarray]    # path -> HxW uint8 of train ids (255 = ignore)
    img_size: Tuple[int, int]                 # default (H, W)
    group_names: Dict[int, str] = field(default_factory=dict)
    group_order: Optional[List[int]] = None   # order used to chunk groups into "natural" clients

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class SegDataset(Dataset):
    """Loads (image, mask) pairs lazily, resizes, augments and normalises them."""

    def __init__(self, samples: Sequence[Sample], load_mask: Callable[[str], np.ndarray],
                 img_size: Tuple[int, int], train: bool = False, augment: bool = True):
        self.samples = list(samples)
        self.load_mask = load_mask
        self.h, self.w = img_size
        self.train = train
        self.augment = augment and train
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, s: Sample) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(s.image).convert("RGB")
        mask = Image.fromarray(self.load_mask(s.mask))
        return img, mask

    def __getitem__(self, idx: int):
        img, mask = self._load(self.samples[idx])
        h, w = self.h, self.w
        if self.augment:
            # random zoom-in crop (scale 1.0-1.5) keeps output size fixed for batching
            s = random.uniform(1.0, 1.5)
            sh, sw = int(round(h * s)), int(round(w * s))
            img = img.resize((sw, sh), Image.BILINEAR)
            mask = mask.resize((sw, sh), Image.NEAREST)
            top, left = random.randint(0, sh - h), random.randint(0, sw - w)
            img = img.crop((left, top, left + w, top + h))
            mask = mask.crop((left, top, left + w, top + h))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)   # NEAREST: never invent labels at boundaries

        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
        if self.augment:
            # mild photometric jitter (brightness / contrast)
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            m = x.mean()
            x = ((x - m) * c + m) * b
            x = x.clamp_(0, 1)
        x = (x - self.mean) / self.std
        y = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()
        return x, y


# --------------------------------------------------------------------------- label statistics
def label_histograms(samples: Sequence[Sample], load_mask: Callable[[str], np.ndarray],
                     num_classes: int, size: int = 96) -> np.ndarray:
    """Per-sample class pixel histogram (N, C) computed on down-sampled masks (cheap)."""
    out = np.zeros((len(samples), num_classes), dtype=np.float64)
    for i, s in enumerate(samples):
        m = Image.fromarray(load_mask(s.mask)).resize((size, size), Image.NEAREST)
        m = np.asarray(m).ravel()
        m = m[m < num_classes]
        out[i] = np.bincount(m, minlength=num_classes)[:num_classes]
    return out


def class_weights_from_hist(hist: np.ndarray, kind: str) -> Optional[torch.Tensor]:
    """hist: (C,) total pixel counts. 'median_freq' follows Eigen & Fergus / SegNet."""
    if kind == "none":
        return None
    freq = hist / max(hist.sum(), 1.0)
    freq = np.where(freq > 0, freq, np.nan)
    if kind == "median_freq":
        w = np.nanmedian(freq) / freq
    elif kind == "inv_sqrt":
        w = 1.0 / np.sqrt(freq)
        w = w / np.nanmean(w)
    else:
        raise ValueError(kind)
    w = np.nan_to_num(w, nan=0.0)
    w = np.clip(w, 0.0, 10.0)
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- download helpers
def download_url(url: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst):
        return dst
    tmp = dst + ".part"
    print(f"downloading {url} -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "fedseg-downloader"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dst)
    return dst


def extract(archive: str, dst_dir: str) -> None:
    print(f"extracting {archive}")
    dst_real = os.path.realpath(dst_dir)
    if archive.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            for m in t.getmembers():   # refuse path traversal
                if not os.path.realpath(os.path.join(dst_dir, m.name)).startswith(dst_real):
                    raise RuntimeError(f"unsafe path in archive: {m.name}")
            t.extractall(dst_dir)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if not os.path.realpath(os.path.join(dst_dir, n)).startswith(dst_real):
                    raise RuntimeError(f"unsafe path in archive: {n}")
            z.extractall(dst_dir)
    else:
        raise ValueError(archive)
