"""CamVid, 11-class benchmark split used by SegNet (367 train / 101 val / 233 test, 360x480).

A random split of the 701 *video frames* would leak near-identical neighbouring frames between
train and test, so the standard sequence-disjoint split is used.

Natural federated clients = the camera sequence a frame comes from (one "car"/drive per client):
  0001TP (dusk, 62 train) | 0006R0 (101) | 0016E5 (204)  -> feature-shift non-IID.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image

from .common import IGNORE_INDEX, DataBundle, Sample, download_url

CLASSES = ["Sky", "Building", "Pole", "Road", "Sidewalk", "Tree", "SignSymbol",
           "Fence", "Car", "Pedestrian", "Bicyclist"]
PALETTE = np.array([[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
                    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
                    [64, 0, 128], [64, 64, 0], [0, 128, 192]], dtype=np.uint8)
SEQUENCES = ["0001TP", "0006R0", "0016E5", "Seq05VD"]
REPO = "https://github.com/alexgkendall/SegNet-Tutorial.git"
ZIP_URL = "https://github.com/alexgkendall/SegNet-Tutorial/archive/refs/heads/master.zip"


def _download(root: str) -> str:
    target = os.path.join(root, "CamVid")
    if os.path.isdir(os.path.join(target, "trainannot")):
        return target
    os.makedirs(root, exist_ok=True)
    repo_dir = os.path.join(root, "SegNet-Tutorial")
    try:  # sparse checkout: only fetches the CamVid folder (~100 MB)
        if not os.path.isdir(repo_dir):
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                            REPO, repo_dir], check=True)
        subprocess.run(["git", "-C", repo_dir, "sparse-checkout", "set", "CamVid"], check=True)
        shutil.move(os.path.join(repo_dir, "CamVid"), target)
        shutil.rmtree(repo_dir, ignore_errors=True)
    except Exception as e:  # git missing / blocked: fall back to the GitHub zip (~350 MB)
        print(f"git sparse checkout failed ({e}); downloading zip archive instead")
        zpath = download_url(ZIP_URL, os.path.join(root, "segnet_tutorial.zip"))
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                parts = n.split("/", 1)
                if len(parts) == 2 and parts[1].startswith("CamVid/") and not n.endswith("/"):
                    out = os.path.join(root, parts[1])
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with z.open(n) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(zpath)
    return target


def load_mask(path: str) -> np.ndarray:
    m = np.array(Image.open(path), dtype=np.uint8)
    m[m >= len(CLASSES)] = IGNORE_INDEX          # 11 = "Unlabelled"/void
    return m


def build(root: str, download: bool = True) -> DataBundle:
    base = os.path.join(root, "CamVid")
    if download:
        base = _download(root)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"CamVid not found in {base}; set --download true")

    def split(name: str):
        out = []
        for img in sorted(glob.glob(os.path.join(base, name, "*.png"))):
            seq = os.path.basename(img).split("_")[0]
            out.append(Sample(img, os.path.join(base, name + "annot", os.path.basename(img)),
                              SEQUENCES.index(seq) if seq in SEQUENCES else len(SEQUENCES)))
        return out

    return DataBundle(
        name="camvid", train=split("train"), val=split("val"), test=split("test"),
        class_names=CLASSES, palette=PALETTE, load_mask=load_mask, img_size=(360, 480),
        group_names={i: s for i, s in enumerate(SEQUENCES)})
