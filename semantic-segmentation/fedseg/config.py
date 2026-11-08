"""Experiment configuration (one dataclass, usable from CLI, notebooks and tests)."""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ALGORITHMS = ("fedavg", "fedprox", "fedadmm", "feddyn", "scaffold", "fednova", "local", "centralized")
BN_MODES = ("avg", "fedbn", "silobn")
SERVER_OPTS = ("sgd", "momentum", "adam")
PARTITIONS = ("iid", "natural", "dirichlet")


@dataclass
class Config:
    # ---------------- experiment ----------------
    name: str = ""                      # run name; auto-generated when empty
    out_dir: str = "runs"
    seed: int = 0
    device: str = "auto"                # auto | cuda | cpu
    amp: bool = True                    # mixed precision on CUDA

    # ---------------- data ----------------
    dataset: str = "camvid"             # camvid | pets | cityscapes | synthetic
    data_root: str = "data"
    download: bool = True
    img_size: Optional[Tuple[int, int]] = None   # (H, W); None = dataset default
    augment: bool = True
    num_workers: int = 2
    max_train_samples: int = 0          # >0: subsample (for smoke tests)
    max_eval_samples: int = 0

    # ---------------- federation ----------------
    num_clients: int = 3
    partition: str = "natural"          # iid | natural (camera sequence / city / breed group) | dirichlet
    dirichlet_alpha: float = 0.5
    clients_per_round: int = 0          # 0 = all clients participate
    rounds: int = 30
    local_epochs: int = 2
    hetero_epochs: bool = False         # each client draws local epochs ~ U{1..local_epochs} (system heterogeneity)
    max_local_steps: int = 0            # >0: cap local SGD steps per round (smoke tests)

    # ---------------- algorithm ----------------
    algorithm: str = "fedavg"
    bn_mode: str = "avg"                # avg (FedAvg) | fedbn (all BN local) | silobn (BN stats local)
    mu: float = 0.01                    # FedProx proximal coefficient
    rho: float = 0.01                   # FedADMM penalty
    alpha: float = 0.01                 # FedDyn regulariser
    server_opt: str = "sgd"             # sgd (lr=1 -> plain averaging) | momentum (FedAvgM) | adam (FedAdam)
    server_lr: float = 1.0
    server_momentum: float = 0.9
    weighted_agg: bool = True           # weight clients by #samples (McMahan et al.)

    # ---------------- model / optimisation ----------------
    model: str = "segnet"               # segnet | unet | deeplabv3_mbv3 | lraspp_mbv3
    pretrained: bool = True             # ImageNet backbone for torchvision models
    norm: str = "bn"                    # bn | gn (U-Net only; GroupNorm is FL-friendly, Hsieh et al. 2020)
    unet_base: int = 32
    batch_size: int = 4
    optimizer: str = "sgd"              # sgd | adam
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: str = "cosine"         # cosine | const  (decayed over communication rounds)
    class_weights: str = "none"         # none | median_freq (SegNet paper) | inv_sqrt
    label_smoothing: float = 0.0

    # ---------------- evaluation / logging ----------------
    eval_every: int = 1
    eval_personalized: bool = False     # also evaluate each client's personalised model (fedbn/silobn/local)
    save_checkpoints: bool = True
    resume: bool = False
    num_vis: int = 6
    verbose: bool = True

    # ------------------------------------------------------------------
    def finalize(self) -> "Config":
        assert self.algorithm in ALGORITHMS, f"unknown algorithm {self.algorithm}"
        assert self.bn_mode in BN_MODES, f"unknown bn_mode {self.bn_mode}"
        assert self.server_opt in SERVER_OPTS, f"unknown server_opt {self.server_opt}"
        assert self.partition in PARTITIONS, f"unknown partition {self.partition}"
        if self.algorithm == "centralized":
            # pooled data == a single client holding everything
            self.num_clients, self.partition, self.clients_per_round = 1, "iid", 0
            self.bn_mode, self.hetero_epochs = "avg", False
        if self.algorithm == "feddyn" and (self.server_opt != "sgd" or self.server_lr != 1.0):
            raise ValueError("FedDyn defines its own server step; use server_opt=sgd, server_lr=1")
        if self.img_size is not None:
            self.img_size = tuple(int(v) for v in self.img_size)
        if not self.name:
            parts = [self.dataset, self.model, self.algorithm]
            if self.algorithm != "centralized":
                parts += [f"{self.partition}{self.num_clients}"]
                if self.bn_mode != "avg":
                    parts.append(self.bn_mode)
                if self.server_opt != "sgd":
                    parts.append(f"srv{self.server_opt}")
            parts.append(f"s{self.seed}")
            self.name = "_".join(parts)
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "y", "t"):
        return True
    if v.lower() in ("0", "false", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Federated semantic segmentation",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    choices = {"algorithm": ALGORITHMS, "bn_mode": BN_MODES, "server_opt": SERVER_OPTS,
               "partition": PARTITIONS}
    for f in dataclasses.fields(Config):
        default = f.default if f.default is not dataclasses.MISSING else None
        flag = "--" + f.name
        if f.name == "img_size":
            p.add_argument(flag, type=int, nargs=2, default=None, metavar=("H", "W"))
        elif isinstance(default, bool):
            p.add_argument(flag, type=_str2bool, default=default, metavar="BOOL")
        elif f.name in choices:
            p.add_argument(flag, type=str, default=default, choices=choices[f.name])
        else:
            p.add_argument(flag, type=type(default) if default is not None else str, default=default)
    return p


def parse_args(argv: Optional[List[str]] = None) -> Config:
    ns = build_parser().parse_args(argv)
    return Config(**vars(ns)).finalize()
