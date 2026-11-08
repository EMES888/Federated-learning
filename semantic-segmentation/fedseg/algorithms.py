"""Federated optimisation strategies.

Every strategy works on the list of *shared* trainable tensors (BN layers can be kept local, see
`bn_mode`) and implements three hooks:

    correct_grad(cid, params, theta)   -> added to the local gradient at every SGD step
    after_local(cid, w, theta, info)   -> client-side state update, returns the upload message
    server_step(theta, msgs, weights)  -> new global tensors (via a pseudo-gradient + server optimiser)

Notation: theta = global model received at the start of the round, w = client model after local
training, p_i = aggregation weight of client i (proportional to #samples, normalised over the
participating clients).

References
  FedAvg   McMahan et al., AISTATS 2017       FedProx  Li et al., MLSys 2020
  FedADMM  Gong, Li & Freris, ICDE 2022       FedDyn   Acar et al., ICLR 2021
  SCAFFOLD Karimireddy et al., ICML 2020      FedNova  Wang et al., NeurIPS 2020
  FedAvgM  Hsu et al., 2019                   FedAdam  Reddi et al., ICLR 2021
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

Tensors = List[torch.Tensor]


def _zeros_like(ts: Tensors, device="cpu") -> Tensors:
    return [torch.zeros_like(t, device=device) for t in ts]


def _to(ts: Tensors, device) -> Tensors:
    return [t.to(device, non_blocking=True) for t in ts]


# ----------------------------------------------------------------------------- server optimisers
class ServerOptimizer:
    """theta <- theta - lr * update(pseudo_grad). sgd with lr=1 reproduces plain averaging."""

    def __init__(self, kind: str, lr: float, momentum: float = 0.9, betas=(0.9, 0.99), tau: float = 1e-3):
        self.kind, self.lr, self.momentum, self.betas, self.tau = kind, lr, momentum, betas, tau
        self.m: Optional[Tensors] = None
        self.v: Optional[Tensors] = None

    @torch.no_grad()
    def step(self, theta: Tensors, grad: Tensors) -> Tensors:
        if self.kind == "sgd":
            return [t - self.lr * g for t, g in zip(theta, grad)]
        if self.kind == "momentum":           # FedAvgM
            if self.m is None:
                self.m = _zeros_like(grad, grad[0].device)
            self.m = [self.momentum * m + g for m, g in zip(self.m, grad)]
            return [t - self.lr * m for t, m in zip(theta, self.m)]
        if self.kind == "adam":               # FedAdam (no bias correction, as in Reddi et al.)
            b1, b2 = self.betas
            if self.m is None:
                self.m = _zeros_like(grad, grad[0].device)
                self.v = [torch.full_like(g, self.tau ** 2) for g in grad]
            self.m = [b1 * m + (1 - b1) * g for m, g in zip(self.m, grad)]
            self.v = [b2 * v + (1 - b2) * g * g for v, g in zip(self.v, grad)]
            return [t - self.lr * m / (v.sqrt() + self.tau) for t, m, v in zip(theta, self.m, self.v)]
        raise ValueError(self.kind)

    def state_dict(self):
        return {"m": self.m, "v": self.v}

    def load_state_dict(self, s):
        self.m, self.v = s["m"], s["v"]


# ----------------------------------------------------------------------------- strategies
class Strategy:
    name = "base"
    upload_factor = 1          # how many model-sized tensors a client uploads per round

    def __init__(self, cfg, num_clients: int, server_opt: ServerOptimizer, state_device="cpu"):
        self.cfg = cfg
        self.num_clients = num_clients
        self.server_opt = server_opt
        self.state_device = state_device
        self.client_state: Dict[int, Dict[str, Tensors]] = {}

    # --- client side -----------------------------------------------------------------------
    def begin_local(self, cid: int, theta: Tensors) -> None:
        """Fetch/initialise the client's persistent state before local training."""

    def correct_grad(self, cid: int, params: Tensors, theta: Tensors) -> None:
        """Modify p.grad in place (called after unscaling, before optimizer.step)."""

    def after_local(self, cid: int, w: Tensors, theta: Tensors, info: Dict) -> Dict:
        return {"w": w}

    # --- server side -----------------------------------------------------------------------
    def server_step(self, theta: Tensors, msgs: List[Dict], p: List[float]) -> Tensors:
        avg = [sum(pi * m["w"][j] for pi, m in zip(p, msgs)) for j in range(len(theta))]
        pseudo_grad = [t - a for t, a in zip(theta, avg)]
        return self.server_opt.step(theta, pseudo_grad)

    def state_dict(self):
        return {"client_state": self.client_state, "server_opt": self.server_opt.state_dict()}

    def load_state_dict(self, s):
        self.client_state = s["client_state"]
        self.server_opt.load_state_dict(s["server_opt"])


class FedAvg(Strategy):
    name = "fedavg"


class FedProx(Strategy):
    """Local objective f_i(w) + mu/2 ||w - theta||^2, applied directly to the gradients."""
    name = "fedprox"

    @torch.no_grad()
    def correct_grad(self, cid, params, theta):
        mu = self.cfg.mu
        for p, t in zip(params, theta):
            if p.grad is not None:
                p.grad.add_(p.detach() - t, alpha=mu)


class FedADMM(Strategy):
    """Augmented Lagrangian  f_i(w) + <y_i, w - theta> + rho/2 ||w - theta||^2.
    Dual ascent  y_i <- y_i + rho (w_i - theta).  Clients upload the change of their augmented
    model u_i = w_i + y_i / rho, server: theta <- theta + eta * sum_i p_i * delta_u_i."""
    name = "fedadmm"

    def _state(self, cid, theta):
        # lazy init at first participation: y=0, u=theta (identical to the paper with full
        # participation; with partial participation it avoids injecting theta^t - theta^0 drift)
        if cid not in self.client_state:
            self.client_state[cid] = {"y": _zeros_like(theta, self.state_device),
                                      "u": [t.detach().to(self.state_device).clone() for t in theta]}
        return self.client_state[cid]

    @torch.no_grad()
    def correct_grad(self, cid, params, theta):
        rho = self.cfg.rho
        y = self._cache_y
        for p, t, yi in zip(params, theta, y):
            if p.grad is not None:
                p.grad.add_(yi).add_(p.detach() - t, alpha=rho)

    def begin_local(self, cid, theta):
        self._cache_y = _to(self._state(cid, theta)["y"], theta[0].device)

    @torch.no_grad()
    def after_local(self, cid, w, theta, info):
        rho = self.cfg.rho
        st = self._state(cid, theta)
        y_new = [y + rho * (wi - t) for y, wi, t in zip(self._cache_y, w, theta)]
        u_new = [wi + y / rho for wi, y in zip(w, y_new)]
        u_old = _to(st["u"], w[0].device)
        delta = [a - b for a, b in zip(u_new, u_old)]
        st["y"] = _to(y_new, self.state_device)
        st["u"] = _to(u_new, self.state_device)
        return {"delta": delta}

    def server_step(self, theta, msgs, p):
        agg = [sum(pi * m["delta"][j] for pi, m in zip(p, msgs)) for j in range(len(theta))]
        return self.server_opt.step(theta, [-a for a in agg])


class FedDyn(Strategy):
    """Dynamic regularisation: f_i(w) - <g_i, w> + alpha/2 ||w - theta||^2,
    g_i <- g_i - alpha (w_i - theta);  h <- h - alpha/m sum_{i in S} (w_i - theta);
    theta <- avg_S(w_i) - h / alpha."""
    name = "feddyn"

    def begin_local(self, cid, theta):
        if cid not in self.client_state:
            self.client_state[cid] = {"g": _zeros_like(theta, self.state_device)}
        self._cache_g = _to(self.client_state[cid]["g"], theta[0].device)

    @torch.no_grad()
    def correct_grad(self, cid, params, theta):
        a = self.cfg.alpha
        for p, t, g in zip(params, theta, self._cache_g):
            if p.grad is not None:
                p.grad.sub_(g).add_(p.detach() - t, alpha=a)

    @torch.no_grad()
    def after_local(self, cid, w, theta, info):
        a = self.cfg.alpha
        g_new = [g - a * (wi - t) for g, wi, t in zip(self._cache_g, w, theta)]
        self.client_state[cid]["g"] = _to(g_new, self.state_device)
        return {"w": w}

    @torch.no_grad()
    def server_step(self, theta, msgs, p):
        a, m = self.cfg.alpha, self.num_clients
        if not hasattr(self, "h"):
            self.h = _zeros_like(theta, theta[0].device)
        avg = [sum(pi * mm["w"][j] for pi, mm in zip(p, msgs)) for j in range(len(theta))]
        drift = [sum(mm["w"][j] - theta[j] for mm in msgs) for j in range(len(theta))]
        self.h = [h - (a / m) * d for h, d in zip(self.h, drift)]
        return [av - h / a for av, h in zip(avg, self.h)]

    def state_dict(self):
        s = super().state_dict()
        s["h"] = getattr(self, "h", None)
        return s

    def load_state_dict(self, s):
        super().load_state_dict(s)
        if s.get("h") is not None:
            self.h = s["h"]


class Scaffold(Strategy):
    """Control variates (option II): local grad += c - c_i;
    c_i+ = c_i - c + (theta - w) / (K * lr_eff);  c <- c + |S|/N * mean_S(c_i+ - c_i)."""
    name = "scaffold"
    upload_factor = 2

    def begin_local(self, cid, theta):
        if not hasattr(self, "c"):
            self.c = _zeros_like(theta, theta[0].device)
        if cid not in self.client_state:
            self.client_state[cid] = {"c": _zeros_like(theta, self.state_device)}
        self._cache_ci = _to(self.client_state[cid]["c"], theta[0].device)

    @torch.no_grad()
    def correct_grad(self, cid, params, theta):
        for p, c, ci in zip(params, self.c, self._cache_ci):
            if p.grad is not None:
                p.grad.add_(c - ci)

    @torch.no_grad()
    def after_local(self, cid, w, theta, info):
        k = max(info["steps"], 1)
        lr = info["lr"]
        if self.cfg.optimizer == "sgd" and self.cfg.momentum > 0:
            lr = lr / (1.0 - self.cfg.momentum)       # effective step size of heavy-ball SGD
        ci_new = [ci - c + (t - wi) / (k * lr) for ci, c, t, wi in zip(self._cache_ci, self.c, theta, w)]
        dc = [a - b for a, b in zip(ci_new, self._cache_ci)]
        self.client_state[cid]["c"] = _to(ci_new, self.state_device)
        return {"w": w, "dc": dc}

    @torch.no_grad()
    def server_step(self, theta, msgs, p):
        n = self.num_clients
        self.c = [c + sum(m["dc"][j] for m in msgs) / n for j, c in enumerate(self.c)]
        return super().server_step(theta, msgs, p)

    def state_dict(self):
        s = super().state_dict()
        s["c"] = getattr(self, "c", None)
        return s

    def load_state_dict(self, s):
        super().load_state_dict(s)
        if s.get("c") is not None:
            self.c = s["c"]


class FedNova(Strategy):
    """Normalised averaging: removes the objective inconsistency caused by clients doing different
    numbers of local steps (relevant with --hetero_epochs)."""
    name = "fednova"

    def _a(self, tau: int) -> float:
        b = self.cfg.momentum if self.cfg.optimizer == "sgd" else 0.0
        if b == 0:
            return float(tau)
        return (tau - b * (1 - b ** tau) / (1 - b)) / (1 - b)

    @torch.no_grad()
    def after_local(self, cid, w, theta, info):
        a = self._a(max(info["steps"], 1))
        return {"d": [(t - wi) / a for t, wi in zip(theta, w)], "a": a}

    @torch.no_grad()
    def server_step(self, theta, msgs, p):
        tau_eff = sum(pi * m["a"] for pi, m in zip(p, msgs))
        g = [tau_eff * sum(pi * m["d"][j] for pi, m in zip(p, msgs)) for j in range(len(theta))]
        return self.server_opt.step(theta, g)


class LocalOnly(Strategy):
    """No communication: each client trains its own model (lower bound / personalisation baseline)."""
    name = "local"
    upload_factor = 0


STRATEGIES = {"fedavg": FedAvg, "centralized": FedAvg, "fedprox": FedProx, "fedadmm": FedADMM,
              "feddyn": FedDyn, "scaffold": Scaffold, "fednova": FedNova, "local": LocalOnly}


def build_strategy(cfg, num_clients: int, state_device="cpu") -> Strategy:
    srv = ServerOptimizer(cfg.server_opt, cfg.server_lr, cfg.server_momentum)
    return STRATEGIES[cfg.algorithm](cfg, num_clients, srv, state_device)
