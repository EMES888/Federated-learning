"""Split the training set across clients.

* iid       : uniform random split.
* natural   : split along the dataset's natural grouping (CamVid sequence, Cityscapes city,
              Pet breed). Groups are chunked contiguously (balanced by #images) when there are
              more groups than clients; groups are subdivided when there are fewer.
* dirichlet : per-group client proportions ~ Dir(alpha) (Hsu et al., 2019); small alpha = more skew.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def iid(n: int, num_clients: int, rng: np.random.RandomState) -> List[List[int]]:
    perm = rng.permutation(n)
    return [sorted(p.tolist()) for p in np.array_split(perm, num_clients)]


def natural(groups: Sequence[int], num_clients: int, rng: np.random.RandomState,
            order: Optional[Sequence[int]] = None) -> List[List[int]]:
    groups = np.asarray(groups)
    present = sorted(set(groups.tolist()))
    if order is not None:
        present = [g for g in order if g in set(present)] + [g for g in present if g not in set(order)]
    members: Dict[int, np.ndarray] = {g: np.where(groups == g)[0] for g in present}
    sizes = np.array([len(members[g]) for g in present], dtype=float)

    if len(present) >= num_clients:
        # contiguous chunks of groups with ~equal #images, each chunk non-empty
        out: List[List[int]] = []
        start = 0
        for c in range(num_clients):
            remaining_clients = num_clients - c
            remaining = sizes[start:].sum()
            target = remaining / remaining_clients
            end, acc = start, 0.0
            while end < len(present) - (remaining_clients - 1):
                if acc > 0 and acc + sizes[end] / 2 > target:
                    break
                acc += sizes[end]
                end += 1
            out.append(sorted(np.concatenate([members[g] for g in present[start:end]]).tolist()))
            start = end
        return out

    # fewer groups than clients: give each group >=1 client, extra clients to the largest groups
    alloc = np.ones(len(present), dtype=int)
    for _ in range(num_clients - len(present)):
        alloc[np.argmax(sizes / alloc)] += 1
    out = []
    for g, k in zip(present, alloc):
        idx = rng.permutation(members[g])
        out += [sorted(p.tolist()) for p in np.array_split(idx, k)]
    return out


def dirichlet(groups: Sequence[int], num_clients: int, alpha: float, rng: np.random.RandomState,
              min_size: int = 2, max_tries: int = 1000) -> List[List[int]]:
    groups = np.asarray(groups)
    for _ in range(max_tries):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for g in np.unique(groups):
            idx = rng.permutation(np.where(groups == g)[0])
            p = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for c, part in enumerate(np.split(idx, cuts)):
                buckets[c] += part.tolist()
        if min(len(b) for b in buckets) >= min_size:
            return [sorted(b) for b in buckets]
    raise RuntimeError("could not find a Dirichlet split with enough samples per client; "
                       "increase alpha or reduce num_clients")


def make_partition(kind: str, groups: Sequence[int], num_clients: int, seed: int,
                   alpha: float = 0.5, order: Optional[Sequence[int]] = None,
                   min_size: int = 2) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    if num_clients == 1:
        return [list(range(len(groups)))]
    if kind == "iid":
        return iid(len(groups), num_clients, rng)
    if kind == "natural":
        return natural(groups, num_clients, rng, order)
    if kind == "dirichlet":
        return dirichlet(groups, num_clients, alpha, rng, min_size)
    raise ValueError(kind)


def heterogeneity(client_hists: np.ndarray) -> Dict[str, float]:
    """Label-distribution skew: mean Jensen-Shannon divergence (base 2, in [0,1]) between each
    client's pixel-class distribution and the pooled one, plus mean total-variation distance."""
    eps = 1e-12
    P = client_hists / np.maximum(client_hists.sum(1, keepdims=True), eps)
    Q = client_hists.sum(0) / max(client_hists.sum(), eps)

    def kl(a, b):
        m = a > 0
        return float(np.sum(a[m] * np.log2(a[m] / np.maximum(b[m], eps))))

    js = [0.5 * kl(p, (p + Q) / 2) + 0.5 * kl(Q, (p + Q) / 2) for p in P]
    tv = [0.5 * float(np.abs(p - Q).sum()) for p in P]
    return {"mean_js": float(np.mean(js)), "mean_tv": float(np.mean(tv))}
