"""Cityscapes (19 train classes). Requires a free account, so it is *not* downloaded automatically:
download leftImg8bit_trainvaltest.zip + gtFine_trainvaltest.zip from https://www.cityscapes-dataset.com
(or `pip install cityscapesscripts && csDownload ...`) and unzip both into <data_root>/cityscapes.

Natural clients = cities (18 training cities), as in the FedDrive benchmark (Fantauzzo et al., 2022).
The official val split (3 unseen cities) is used as the test set; 5% of train is held out for validation.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

from .common import IGNORE_INDEX, DataBundle, Sample

CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle"]
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)
# labelId -> trainId (official cityscapesscripts table)
_ID2TRAIN = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _lid, _tid in {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                   24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}.items():
    _ID2TRAIN[_lid] = _tid


def load_mask(path: str) -> np.ndarray:
    return _ID2TRAIN[np.array(Image.open(path), dtype=np.uint8)]


def build(root: str, download: bool = False, seed: int = 0, val_fraction: float = 0.05) -> DataBundle:
    base = os.path.join(root, "cityscapes")
    if not os.path.isdir(os.path.join(base, "leftImg8bit")):
        raise FileNotFoundError(
            f"Cityscapes not found under {base}. It needs registration; see fedseg/data/cityscapes.py")
    cities = sorted(os.listdir(os.path.join(base, "leftImg8bit", "train")))

    def split(name):
        out = []
        for img in sorted(glob.glob(os.path.join(base, "leftImg8bit", name, "*", "*_leftImg8bit.png"))):
            city = os.path.basename(os.path.dirname(img))
            stem = os.path.basename(img).replace("_leftImg8bit.png", "")
            mask = os.path.join(base, "gtFine", name, city, stem + "_gtFine_labelIds.png")
            out.append(Sample(img, mask, cities.index(city) if city in cities else -1))
        return out

    train = split("train")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train))
    k = int(len(train) * val_fraction)
    val = [train[i] for i in perm[:k]]
    train = [train[i] for i in sorted(perm[k:])]
    return DataBundle(name="cityscapes", train=train, val=val, test=split("val"),
                      class_names=CLASSES, palette=PALETTE, load_mask=load_mask,
                      img_size=(256, 512), group_names=dict(enumerate(cities)))
