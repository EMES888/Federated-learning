"""End-to-end smoke tests on the synthetic dataset (CPU, ~1-2 min total)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedseg import Config, run_experiment  # noqa: E402
from fedseg.data import make_partition  # noqa: E402
from fedseg.metrics import ConfusionMatrix  # noqa: E402
from fedseg.models import build_model  # noqa: E402


def tiny_cfg(tmp_path, **kw):
    base = dict(dataset="synthetic", data_root=str(tmp_path / "data"), out_dir=str(tmp_path / "runs"),
                model="unet", unet_base=8, rounds=2, local_epochs=1, num_clients=3, batch_size=4,
                device="cpu", num_workers=0, max_local_steps=3, num_vis=2, verbose=False)
    base.update(kw)
    return Config(**base)


@pytest.mark.parametrize("algorithm", ["fedavg", "fedprox", "fedadmm", "feddyn", "scaffold", "fednova",
                                       "local", "centralized"])
def test_every_algorithm_runs(tmp_path, algorithm):
    res = run_experiment(tiny_cfg(tmp_path, algorithm=algorithm))
    assert 0.0 <= res["test"]["mIoU"] <= 1.0
    out = os.path.join(str(tmp_path / "runs"), res["name"])
    for f in ("results.json", "history.csv", "config.json", "partition.json", "curves.png"):
        assert os.path.exists(os.path.join(out, f)), f


@pytest.mark.parametrize("bn_mode", ["fedbn", "silobn"])
def test_local_batchnorm_modes(tmp_path, bn_mode):
    res = run_experiment(tiny_cfg(tmp_path, bn_mode=bn_mode, eval_personalized=True, model="segnet"))
    assert res["test_personalized"] is not None
    assert len(res["test_personalized"]["per_client_mIoU"]) == 3


@pytest.mark.parametrize("server_opt,lr", [("momentum", 1.0), ("adam", 0.01)])
def test_server_optimizers(tmp_path, server_opt, lr):
    run_experiment(tiny_cfg(tmp_path, server_opt=server_opt, server_lr=lr))


def test_partial_participation_and_hetero_epochs(tmp_path):
    res = run_experiment(tiny_cfg(tmp_path, num_clients=4, clients_per_round=2, hetero_epochs=True,
                                  local_epochs=2, algorithm="fedadmm", partition="dirichlet",
                                  dirichlet_alpha=1.0))
    hist = open(os.path.join(str(tmp_path / "runs"), res["name"], "history.csv")).read()
    assert hist.count("\n") == 3


def test_resume_matches_history(tmp_path):
    cfg = tiny_cfg(tmp_path, rounds=2, name="resume_run")
    run_experiment(cfg)
    cfg2 = tiny_cfg(tmp_path, rounds=3, name="resume_run", resume=True)
    run_experiment(cfg2)
    hist = open(os.path.join(str(tmp_path / "runs"), "resume_run", "history.csv")).read().strip().split("\n")
    assert len(hist) == 1 + 3


def test_fedavg_one_client_equals_centralised_sgd(tmp_path):
    """Sanity: FedAvg with a single client and server lr 1 must leave the client model unchanged."""
    from fedseg.trainer import FederatedTrainer
    tr = FederatedTrainer(tiny_cfg(tmp_path, algorithm="fedavg", num_clients=1, augment=False, rounds=1))
    msg = tr.train_client(0, 0, 0.01, 1)
    w_client = [t.clone() for t in msg["w"]]
    tr.aggregate([0], [msg])
    named = dict(tr.global_model.named_parameters())
    for n, w in zip(tr.shared_names, w_client):
        assert torch.allclose(named[n], w, atol=1e-6)


@pytest.mark.parametrize("name", ["segnet", "unet"])
@pytest.mark.parametrize("hw", [(64, 64), (90, 120), (360, 480)])
def test_models_any_size(name, hw):
    m = build_model(name, 5, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, *hw))
    assert out.shape == (1, 5, *hw)


def test_partitions_cover_all_samples():
    groups = np.repeat(np.arange(5), [10, 20, 30, 5, 35])
    for kind in ("iid", "natural", "dirichlet"):
        for n in (2, 3, 7):
            parts = make_partition(kind, groups, n, seed=0, alpha=0.5)
            flat = sorted(i for p in parts for i in p)
            assert flat == list(range(len(groups))), (kind, n)
            assert len(parts) == n


def test_miou_ignores_void():
    cm = ConfusionMatrix(3, ignore_index=255)
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 1] = 1.0                                   # predicts class 1 everywhere
    target = torch.tensor([[[1, 1], [255, 255]]])        # void pixels must not count
    cm.update(logits, target)
    r = cm.compute()
    assert r["mIoU"] == 1.0 and r["pixel_acc"] == 1.0
