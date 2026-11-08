"""Oxford-IIIT Pet (Parkhi et al., 2012) as a 3-class segmentation task: background / cat / dog.

Pixel labels come from the trimaps (1 = pet, 2 = background, 3 = boundary -> ignored); the pet
pixels are labelled with the species, so a client that only owns dog breeds never sees the "cat"
class -> genuine *label-skew* non-IID, complementary to CamVid's feature (appearance) shift.

Natural groups = the 37 breeds (ordered cats first, then dogs), chunked into clients.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .common import IGNORE_INDEX, DataBundle, Sample, download_url, extract

CLASSES = ["background", "cat", "dog"]
PALETTE = np.array([[0, 0, 0], [230, 120, 30], [40, 120, 230]], dtype=np.uint8)
URLS = {"images": "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz",
        "annotations": "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz"}



def _species(name: str) -> int:
    """1 = cat, 2 = dog. In this dataset cat-breed file names are Capitalised, dog-breed ones not
    (checked against annotations/list.txt in tests); stateless so it is safe in DataLoader workers."""
    return 1 if name[:1].isupper() else 2


def _download(root: str) -> str:
    base = os.path.join(root, "oxford-iiit-pet")
    if os.path.isfile(os.path.join(base, "annotations", "trainval.txt")) and \
            os.path.isdir(os.path.join(base, "images")):
        return base
    os.makedirs(base, exist_ok=True)
    for key, url in URLS.items():
        arc = download_url(url, os.path.join(base, f"{key}.tar.gz"))
        extract(arc, base)
        os.remove(arc)
    return base


def load_mask(path: str) -> np.ndarray:
    tri = np.array(Image.open(path), dtype=np.uint8)
    species = _species(os.path.basename(path))
    out = np.full(tri.shape, IGNORE_INDEX, dtype=np.uint8)
    out[tri == 2] = 0
    out[tri == 1] = species
    return out


def build(root: str, download: bool = True, val_fraction: float = 0.1, seed: int = 0) -> DataBundle:
    base = _download(root) if download else os.path.join(root, "oxford-iiit-pet")
    ann = os.path.join(base, "annotations")
    if not os.path.isdir(ann):
        raise FileNotFoundError(f"Oxford-IIIT Pet not found in {base}; set --download true")

    breed_names, breed_species = {}, {}

    def read(listfile):
        out = []
        with open(os.path.join(ann, listfile)) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                name, cls_id, species, _ = line.split()
                cls_id, species = int(cls_id) - 1, int(species)
                breed_names[cls_id] = name.rsplit("_", 1)[0]
                breed_species[cls_id] = species
                assert species == _species(name), name
                img = os.path.join(base, "images", name + ".jpg")
                if os.path.exists(img):
                    out.append(Sample(img, os.path.join(ann, "trimaps", name + ".png"), cls_id))
        return out

    trainval, test = read("trainval.txt"), read("test.txt")
    # stratified (by breed) train/val split
    rng = np.random.RandomState(seed)
    train, val = [], []
    for g in sorted({s.group for s in trainval}):
        items = [s for s in trainval if s.group == g]
        rng.shuffle(items)
        k = max(1, int(round(len(items) * val_fraction)))
        val += items[:k]
        train += items[k:]
    order = sorted(breed_names, key=lambda b: (breed_species[b], b))   # cats first, then dogs
    return DataBundle(name="pets", train=train, val=val, test=test, class_names=CLASSES,
                      palette=PALETTE, load_mask=load_mask, img_size=(256, 256),
                      group_names=breed_names, group_order=order)
