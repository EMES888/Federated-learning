from __future__ import annotations

from . import camvid, cityscapes, pets, synthetic
from .common import (IGNORE_INDEX, DataBundle, Sample, SegDataset, class_weights_from_hist,
                     label_histograms)
from .partition import heterogeneity, make_partition

DATASETS = {"camvid": camvid.build, "pets": pets.build, "cityscapes": cityscapes.build,
            "synthetic": synthetic.build}


def build_bundle(name: str, root: str, download: bool = True, seed: int = 0) -> DataBundle:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    kwargs = {"download": download}
    if name in ("pets", "cityscapes"):
        kwargs["seed"] = seed
    return DATASETS[name](root, **kwargs)


__all__ = ["IGNORE_INDEX", "DataBundle", "Sample", "SegDataset", "build_bundle", "make_partition",
           "heterogeneity", "label_histograms", "class_weights_from_hist", "DATASETS"]
