"""Algorithm-level correctness on a heterogeneous quadratic problem with a known optimum.

Client i: f_i(w) = h_i/2 ||w - a_i||^2   (different curvature + different optimum = non-IID)
Global objective sum_i p_i f_i  ->  w* = sum p_i h_i a_i / sum p_i h_i.
With many local steps FedAvg drifts to a biased fixed point (client drift); FedProx reduces the bias;
FedADMM / FedDyn / SCAFFOLD should converge to w* (that is the point of these methods).
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedseg.algorithms import build_strategy  # noqa: E402
from fedseg.config import Config  # noqa: E402

H = torch.tensor([1.0, 4.0, 0.5])
A = torch.tensor([[2.0, -1.0], [-1.0, 3.0], [4.0, 0.0]])
P = [1 / 3] * 3
W_STAR = (H[:, None] * A).sum(0) / H.sum()


def simulate(algorithm, rounds=300, local_steps=20, lr=0.05, steps_per_client=None, **kw):
    cfg = Config(algorithm=algorithm, optimizer="sgd", momentum=0.0, **kw)
    strat = build_strategy(cfg, num_clients=3)
    theta = [torch.zeros(2)]
    for _ in range(rounds):
        msgs = []
        for cid in range(3):
            K = steps_per_client[cid] if steps_per_client else local_steps
            w = theta[0].clone().requires_grad_(True)
            th = [theta[0].clone()]
            strat.begin_local(cid, th)
            for _ in range(K):
                loss = 0.5 * H[cid] * ((w - A[cid]) ** 2).sum()
                w.grad = None
                loss.backward()
                strat.correct_grad(cid, [w], th)
                with torch.no_grad():
                    w -= lr * w.grad
            msgs.append(strat.after_local(cid, [w.detach().clone()], th, {"steps": K, "lr": lr}))
        theta = strat.server_step(theta, msgs, P)
    return theta[0]


def err(w):
    return float((w - W_STAR).norm())


def test_fedavg_is_biased_under_client_drift():
    assert err(simulate("fedavg")) > 0.3


def test_fedavg_single_local_step_is_unbiased():
    # one local step == centralised gradient descent on the global objective
    assert err(simulate("fedavg", local_steps=1, rounds=2000)) < 1e-3


def test_fedprox_reduces_bias():
    assert err(simulate("fedprox", mu=1.0)) < err(simulate("fedavg"))


@pytest.mark.parametrize("algo,kw", [("fedadmm", {"rho": 1.0}), ("feddyn", {"alpha": 1.0}),
                                     ("scaffold", {})])
def test_drift_correcting_methods_reach_optimum(algo, kw):
    assert err(simulate(algo, **kw)) < 1e-2, algo


def test_fednova_removes_step_imbalance_bias():
    # identical curvature, different #local steps: FedAvg is pulled towards the busy client
    global H
    h_old = H
    H = torch.tensor([1.0, 1.0, 1.0])
    try:
        w_star = A.mean(0)
        steps = [2, 30, 5]
        # FedNova is unbiased to first order in lr*tau (Wang et al. 2020), so use a small step size
        kw = dict(steps_per_client=steps, lr=0.002, rounds=1500)
        e_avg = float((simulate("fedavg", **kw) - w_star).norm())
        e_nova = float((simulate("fednova", **kw) - w_star).norm())
        assert e_avg > 0.5, e_avg
        assert e_nova < 0.06 and e_nova < 0.1 * e_avg, (e_nova, e_avg)
    finally:
        H = h_old


@pytest.mark.parametrize("server_opt,lr", [("momentum", 0.5), ("adam", 0.05)])
def test_server_optimizers_converge(server_opt, lr):
    w = simulate("fedavg", local_steps=1, rounds=1500, server_opt=server_opt, server_lr=lr)
    assert err(w) < 0.05
