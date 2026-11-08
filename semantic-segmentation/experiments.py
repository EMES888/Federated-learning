"""Experiment suites used in the paper-style comparison (importable from the Colab notebook).

Each suite is a list of Config overrides; `run_suite` skips runs that already have results.json
(so a disconnected Colab session can simply re-run the cell) and resumes half-finished ones.
"""
from __future__ import annotations

import os
import traceback
from typing import Dict, List

from fedseg import Config, run_experiment

# ---------------------------------------------------------------------------- budgets
PRESETS = {
    # ~10-25 min per CamVid run on a Colab T4 with deeplabv3_mbv3
    "quick": dict(rounds=20, local_epochs=2),
    "full": dict(rounds=40, local_epochs=2),
}


def camvid_suite(model: str = "deeplabv3_mbv3", img_size=(240, 320)) -> List[Dict]:
    """Natural non-IID: one client per CamVid drive sequence (3 'cars'); full participation."""
    base = dict(dataset="camvid", model=model, img_size=img_size, partition="natural", num_clients=3,
                batch_size=8, lr=0.01, class_weights="inv_sqrt")
    return [
        {**base, "algorithm": "centralized"},                        # upper bound (pooled data)
        {**base, "algorithm": "local", "eval_personalized": True},   # lower bound (no collaboration)
        {**base, "algorithm": "fedavg"},
        {**base, "algorithm": "fedprox", "mu": 0.01},
        {**base, "algorithm": "fedadmm", "rho": 0.01},
        {**base, "algorithm": "feddyn", "alpha": 0.01},
        {**base, "algorithm": "scaffold"},
        {**base, "algorithm": "fednova"},
        {**base, "algorithm": "fedavg", "server_opt": "momentum", "server_lr": 1.0},   # FedAvgM
        {**base, "algorithm": "fedavg", "bn_mode": "fedbn", "eval_personalized": True},
        {**base, "algorithm": "fedavg", "bn_mode": "silobn", "eval_personalized": True},
        {**base, "algorithm": "fedadmm", "rho": 0.01, "bn_mode": "silobn", "eval_personalized": True},
    ]


def camvid_system_hetero_suite(model: str = "deeplabv3_mbv3", img_size=(240, 320)) -> List[Dict]:
    """6 clients (sequences split in two), 3 sampled per round, local epochs ~ U{1..3}:
    the system-heterogeneity regime FedADMM / FedProx / FedNova were designed for."""
    base = dict(dataset="camvid", model=model, img_size=img_size, partition="natural", num_clients=6,
                clients_per_round=3, hetero_epochs=True, local_epochs=3, batch_size=8,
                class_weights="inv_sqrt")
    return [{**base, "algorithm": a, **extra} for a, extra in
            [("fedavg", {}), ("fedprox", {"mu": 0.01}), ("fedadmm", {"rho": 0.01}), ("fednova", {}),
             ("scaffold", {})]]


def pets_suite(model: str = "deeplabv3_mbv3", partition: str = "natural", alpha: float = 0.3) -> List[Dict]:
    """Label-skew non-IID, 6 clients, 3 sampled per round.
    partition='natural': breeds chunked in order -> 2 cat-only + 4 dog-only clients (strong skew,
    pixel-class JS ~0.10). partition='dirichlet': Dir(alpha) over breeds (mild skew, JS ~0.003-0.01)."""
    base = dict(dataset="pets", model=model, img_size=(224, 224), partition=partition,
                dirichlet_alpha=alpha, num_clients=6, clients_per_round=3, local_epochs=1,
                batch_size=16, lr=0.01)
    return [
        {**base, "algorithm": "centralized", "local_epochs": 1},
        {**base, "algorithm": "fedavg"},
        {**base, "algorithm": "fedprox", "mu": 0.01},
        {**base, "algorithm": "fedadmm", "rho": 0.01},
        {**base, "algorithm": "feddyn", "alpha": 0.01},
        {**base, "algorithm": "scaffold"},
        {**base, "algorithm": "fedavg", "server_opt": "momentum"},
        {**base, "algorithm": "fedavg", "bn_mode": "silobn", "eval_personalized": True},
    ]


def sweep(base: Dict, key: str, values) -> List[Dict]:
    return [{**base, key: v, "name": f"{base.get('dataset')}_{base.get('algorithm')}_{key}{v}"} for v in values]


def run_suite(suite: List[Dict], preset: str = "quick", seeds=(0,), common: Dict = None) -> List[Dict]:
    results = []
    for seed in seeds:
        for overrides in suite:
            kw = {**PRESETS[preset], **(common or {}), **overrides, "seed": seed}
            if "name" in kw and seed:
                kw["name"] = f"{kw['name']}_s{seed}"
            cfg = Config(**kw).finalize()
            done = os.path.join(cfg.out_dir, cfg.name, "results.json")
            if os.path.exists(done):
                print(f"[skip] {cfg.name} already finished")
                continue
            cfg.resume = True   # continue a half-finished run (Colab disconnects)
            try:
                results.append(run_experiment(cfg))
            except Exception:
                print(f"[error] {cfg.name} failed:\n{traceback.format_exc()}")
    return results
