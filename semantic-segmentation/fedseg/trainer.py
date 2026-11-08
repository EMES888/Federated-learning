"""Federated training engine."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .algorithms import build_strategy
from .config import Config
from .data import (IGNORE_INDEX, SegDataset, build_bundle, class_weights_from_hist, heterogeneity,
                   label_histograms, make_partition)
from .metrics import ConfusionMatrix
from .models import build_model, count_params, norm_layer_names


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


class FederatedTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg.finalize()
        set_seed(cfg.seed)
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self.amp = cfg.amp and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.out_dir = os.path.join(cfg.out_dir, cfg.name)
        os.makedirs(self.out_dir, exist_ok=True)
        cfg.save(os.path.join(self.out_dir, "config.json"))
        self.log_path = os.path.join(self.out_dir, "log.txt")

        self._build_data()
        self._build_model()
        self.strategy = build_strategy(cfg, self.num_clients, state_device="cpu")
        self.scaler = _grad_scaler(self.amp)
        self.history: List[Dict] = []
        self.best_score, self.best_round, self.start_round = -1.0, -1, 0
        self.comm_mb = 0.0
        if cfg.resume:
            self._load_checkpoint()

    # ------------------------------------------------------------------ setup
    def log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def _build_data(self) -> None:
        cfg = self.cfg
        b = build_bundle(cfg.dataset, cfg.data_root, cfg.download, cfg.seed)
        self.bundle = b
        rng = np.random.RandomState(cfg.seed)
        train, val, test = b.train, b.val, b.test
        if cfg.max_train_samples and len(train) > cfg.max_train_samples:
            train = [train[i] for i in sorted(rng.choice(len(train), cfg.max_train_samples, replace=False))]
        if cfg.max_eval_samples:
            val = val[: cfg.max_eval_samples]
            test = test[: cfg.max_eval_samples]
        self.img_size = tuple(cfg.img_size) if cfg.img_size else b.img_size

        groups = [s.group for s in train]
        self.partition = make_partition(cfg.partition, groups, cfg.num_clients, cfg.seed,
                                        cfg.dirichlet_alpha, b.group_order, min_size=cfg.batch_size)
        self.num_clients = len(self.partition)
        self.client_sizes = [len(p) for p in self.partition]

        train_ds = SegDataset(train, b.load_mask, self.img_size, train=True, augment=cfg.augment)
        self.client_loaders = []
        for cid, idx in enumerate(self.partition):
            g = torch.Generator()
            g.manual_seed(cfg.seed * 1000 + cid)
            self.client_loaders.append(DataLoader(
                Subset(train_ds, idx), batch_size=cfg.batch_size, shuffle=True,
                drop_last=len(idx) > cfg.batch_size,   # avoid size-1 batches (BatchNorm)
                num_workers=cfg.num_workers, pin_memory=self.device.type == "cuda",
                worker_init_fn=_seed_worker, generator=g))
        mk = lambda s: DataLoader(SegDataset(s, b.load_mask, self.img_size, train=False),
                                  batch_size=max(1, cfg.batch_size * 2), shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=self.device.type == "cuda")
        self.val_loader, self.test_loader = mk(val), mk(test)
        self.test_samples = test

        # label statistics -> class weights + heterogeneity report
        hist = label_histograms(train, b.load_mask, b.num_classes)
        client_hist = np.stack([hist[idx].sum(0) for idx in self.partition])
        self.class_weights = class_weights_from_hist(hist.sum(0), cfg.class_weights)
        self.partition_stats = {
            "client_sizes": self.client_sizes,
            "client_groups": [{b.group_names.get(g, str(g)): int(c) for g, c in
                               zip(*np.unique([groups[i] for i in idx], return_counts=True))}
                              for idx in self.partition],
            "client_class_dist": (client_hist / np.maximum(client_hist.sum(1, keepdims=True), 1)).round(4).tolist(),
            **heterogeneity(client_hist),
        }
        with open(os.path.join(self.out_dir, "partition.json"), "w") as f:
            json.dump(self.partition_stats, f, indent=2)
        self.log(f"[data] {b.name}: train={len(train)} val={len(val)} test={len(test)} "
                 f"size={self.img_size} classes={b.num_classes}")
        self.log(f"[data] {cfg.partition} partition over {self.num_clients} clients: sizes={self.client_sizes} "
                 f"JS={self.partition_stats['mean_js']:.3f}")
        for cid, cg in enumerate(self.partition_stats["client_groups"]):
            self.log(f"        client {cid}: {cg}")

    def _build_model(self) -> None:
        cfg = self.cfg
        self.global_model = build_model(cfg.model, self.bundle.num_classes, cfg.pretrained,
                                        cfg.norm, cfg.unet_base).to(self.device)
        self.work_model = copy.deepcopy(self.global_model)
        bn_params, bn_buffers = norm_layer_names(self.global_model)
        if cfg.bn_mode == "fedbn":
            self.local_keys = bn_params | bn_buffers
        elif cfg.bn_mode == "silobn":
            self.local_keys = set(bn_buffers)
        else:
            self.local_keys = set()
        if cfg.bn_mode != "avg" and not (bn_params or bn_buffers):
            self.log(f"[warn] bn_mode={cfg.bn_mode} but the model has no BatchNorm layers")
        self.shared_names = [n for n, _ in self.global_model.named_parameters() if n not in self.local_keys]
        self.shared_buffers = [n for n, _ in self.global_model.named_buffers() if n not in self.local_keys]
        self.local_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.init_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        n_shared = sum(dict(self.global_model.named_parameters())[n].numel() for n in self.shared_names)
        n_buf = sum(dict(self.global_model.named_buffers())[n].numel() for n in self.shared_buffers
                    if not n.endswith("num_batches_tracked"))
        self.model_mb = (n_shared + n_buf) * 4 / 2 ** 20
        w = self.class_weights.to(self.device) if self.class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE_INDEX,
                                             label_smoothing=cfg.label_smoothing)
        self.log(f"[model] {cfg.model}: {count_params(self.global_model) / 1e6:.2f}M params, "
                 f"{self.model_mb:.1f} MB shared per upload, local keys={len(self.local_keys)}")

    # ------------------------------------------------------------------ helpers
    def _lr(self, rnd: int) -> float:
        if self.cfg.lr_schedule == "cosine":
            return self.cfg.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * rnd / max(1, self.cfg.rounds))))
        return self.cfg.lr

    def _make_optimizer(self, model: nn.Module, lr: float):
        cfg = self.cfg
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum,
                                   weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adam":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        raise ValueError(cfg.optimizer)

    def _client_model_state(self, cid: int) -> Dict[str, torch.Tensor]:
        """Global state with the client's local (non-shared) entries filled in."""
        if self.cfg.algorithm == "local":
            return self.local_states.get(cid, self.init_state)
        state = self.global_model.state_dict()
        if self.local_keys:
            local = self.local_states.get(cid)
            if local is None:
                local = {k: self.init_state[k] for k in self.local_keys}
            state = {**state, **local}
        return state

    def _averaged_local_state(self) -> Dict[str, torch.Tensor]:
        """Sample-weighted average of the clients' local BN entries (a deployable global model)."""
        cids = [c for c in self.local_states]
        if not cids:
            return {k: self.init_state[k] for k in self.local_keys}
        w = np.array([self.client_sizes[c] for c in cids], dtype=float)
        w /= w.sum()
        out = {}
        for k in self.local_keys:
            ts = [self.local_states[c][k] for c in cids]
            if ts[0].is_floating_point():
                out[k] = sum(float(wi) * t for wi, t in zip(w, ts))
            else:
                out[k] = max(ts, key=lambda t: int(t))
        return out

    # ------------------------------------------------------------------ client update
    def train_client(self, cid: int, rnd: int, lr: float, epochs: int) -> Dict:
        cfg, m = self.cfg, self.work_model
        m.load_state_dict(self._client_model_state(cid))
        named = dict(m.named_parameters())
        params = [named[n] for n in self.shared_names]
        theta = [p.detach().clone() for p in params]
        self.strategy.begin_local(cid, theta)
        opt = self._make_optimizer(m, lr)
        m.train()
        steps, loss_sum = 0, 0.0
        for _ in range(epochs):
            for x, y in self.client_loaders[cid]:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    out = m(x)
                loss = self.criterion(out.float(), y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(opt)
                self.strategy.correct_grad(cid, params, theta)
                self.scaler.step(opt)
                self.scaler.update()
                steps += 1
                loss_sum += loss.item()
                if cfg.max_local_steps and steps >= cfg.max_local_steps:
                    break
            if cfg.max_local_steps and steps >= cfg.max_local_steps:
                break
        if not math.isfinite(loss_sum):
            raise FloatingPointError(f"client {cid} diverged in round {rnd} (loss={loss_sum}); lower lr / rho")
        w = [p.detach().clone() for p in params]
        msg = self.strategy.after_local(cid, w, theta, {"steps": steps, "lr": lr})
        state = m.state_dict()
        msg["buffers"] = {k: state[k].detach().clone() for k in self.shared_buffers}
        if cfg.algorithm == "local":
            self.local_states[cid] = {k: v.detach().cpu().clone() for k, v in state.items()}
        elif self.local_keys:
            self.local_states[cid] = {k: state[k].detach().cpu().clone() for k in self.local_keys}
        msg["loss"] = loss_sum / max(steps, 1)
        msg["steps"] = steps
        return msg

    # ------------------------------------------------------------------ server update
    @torch.no_grad()
    def aggregate(self, cids: List[int], msgs: List[Dict]) -> None:
        if self.cfg.algorithm == "local":
            return
        sizes = np.array([self.client_sizes[c] for c in cids], dtype=float)
        p = (sizes / sizes.sum()) if self.cfg.weighted_agg else np.full(len(cids), 1.0 / len(cids))
        p = [float(v) for v in p]
        named = dict(self.global_model.named_parameters())
        theta = [named[n].detach().clone() for n in self.shared_names]
        new = self.strategy.server_step(theta, msgs, p)
        for n, t in zip(self.shared_names, new):
            named[n].copy_(t)
        buffers = dict(self.global_model.named_buffers())
        for k in self.shared_buffers:
            vals = [m["buffers"][k] for m in msgs]
            if vals[0].is_floating_point():
                buffers[k].copy_(sum(pi * v for pi, v in zip(p, vals)))
            else:
                buffers[k].copy_(max(vals, key=lambda t: int(t)))

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, loader) -> Dict:
        model.eval()
        cm = ConfusionMatrix(self.bundle.num_classes, IGNORE_INDEX, self.device)
        loss_sum, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                out = model(x)
            out = out.float()
            loss_sum += self.criterion(out, y).item() * x.size(0)
            n += x.size(0)
            cm.update(out, y)
        res = cm.compute(self.bundle.class_names)
        res["loss"] = loss_sum / max(n, 1)
        return res

    def global_eval_state(self) -> Dict[str, torch.Tensor]:
        state = self.global_model.state_dict()
        if self.local_keys:
            state = {**state, **{k: v.to(self.device) for k, v in self._averaged_local_state().items()}}
        return state

    def evaluate(self, loader, personalized: bool = False) -> Dict:
        res: Dict = {}
        if self.cfg.algorithm != "local":
            self.work_model.load_state_dict(self.global_eval_state())
            res["global"] = self.evaluate_model(self.work_model, loader)
        if personalized or self.cfg.algorithm == "local":
            per = []
            for cid in range(self.num_clients):
                self.work_model.load_state_dict(self._client_model_state(cid))
                per.append(self.evaluate_model(self.work_model, loader))
            summary = {k: float(np.mean([r[k] for r in per]))
                       for k in ("mIoU", "pixel_acc", "mean_acc", "mDice", "fwIoU", "loss")}
            summary["per_class_iou"] = {
                c: (float(np.mean(v)) if (v := [r["per_class_iou"][c] for r in per
                                                if r["per_class_iou"][c] is not None]) else None)
                for c in per[0]["per_class_iou"]}
            summary["per_client_mIoU"] = [r["mIoU"] for r in per]
            summary["clients"] = per
            res["personalized"] = summary
        res["main"] = res.get("global", res.get("personalized"))
        return res

    # ------------------------------------------------------------------ checkpointing
    def _ckpt_payload(self, rnd: int) -> Dict:
        return {"round": rnd, "global": self.global_model.state_dict(), "local_states": self.local_states,
                "strategy": self.strategy.state_dict(), "history": self.history,
                "best_score": self.best_score, "best_round": self.best_round, "comm_mb": self.comm_mb,
                "scaler": self.scaler.state_dict()}

    def _save_best(self) -> None:
        if self.cfg.algorithm == "local":
            payload = {"clients": self.local_states}
        else:
            payload = {"model": {k: v.cpu() for k, v in self.global_eval_state().items()},
                       "local_states": self.local_states}
        torch.save(payload, os.path.join(self.out_dir, "best.pt"))

    def _load_best(self) -> None:
        path = os.path.join(self.out_dir, "best.pt")
        if not os.path.exists(path):
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if self.cfg.algorithm == "local":
            self.local_states = ck["clients"]
        else:
            # global_eval_state() of the best round already contains the averaged BN entries
            self.global_model.load_state_dict(ck["model"])
            self.local_states = ck["local_states"]

    def _load_checkpoint(self) -> None:
        path = os.path.join(self.out_dir, "last.pt")
        if not os.path.exists(path):
            self.log("[resume] no checkpoint found, starting from scratch")
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.global_model.load_state_dict(ck["global"])
        self.local_states = ck["local_states"]
        st = ck["strategy"]

        def to_dev(o):
            if isinstance(o, torch.Tensor):
                return o.to(self.device)
            if isinstance(o, list):
                return [to_dev(v) for v in o]
            return o
        for k in ("c", "h"):
            if st.get(k) is not None:
                st[k] = to_dev(st[k])
        st["server_opt"] = {k: to_dev(v) for k, v in st["server_opt"].items()}
        self.strategy.load_state_dict(st)
        self.history, self.best_score = ck["history"], ck["best_score"]
        self.best_round, self.comm_mb = ck["best_round"], ck["comm_mb"]
        self.scaler.load_state_dict(ck["scaler"])
        self.start_round = ck["round"] + 1
        self.log(f"[resume] continuing from round {self.start_round}")

    def _write_history(self) -> None:
        if not self.history:
            return
        keys = list(self.history[0].keys())
        with open(os.path.join(self.out_dir, "history.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(self.history)

    # ------------------------------------------------------------------ main loop
    def run(self) -> Dict:
        cfg = self.cfg
        rng = np.random.RandomState(cfg.seed + 1)
        for _ in range(self.start_round):  # keep client sampling identical after resume
            rng.permutation(self.num_clients)
            rng.randint(1, cfg.local_epochs + 1, size=self.num_clients)
        k = cfg.clients_per_round if 0 < cfg.clients_per_round < self.num_clients else self.num_clients
        t_start = time.time()
        for rnd in range(self.start_round, cfg.rounds):
            t0 = time.time()
            lr = self._lr(rnd)
            cids = sorted(rng.permutation(self.num_clients)[:k].tolist())
            ep_draw = rng.randint(1, cfg.local_epochs + 1, size=self.num_clients)
            msgs = []
            for cid in cids:
                epochs = int(ep_draw[cid]) if cfg.hetero_epochs else cfg.local_epochs
                msgs.append(self.train_client(cid, rnd, lr, epochs))
            self.aggregate(cids, msgs)
            self.comm_mb += 2 * len(cids) * self.model_mb * max(self.strategy.upload_factor, 0)
            sz = np.array([self.client_sizes[c] for c in cids], dtype=float)
            train_loss = float(np.sum(sz * np.array([m["loss"] for m in msgs])) / sz.sum())
            row = {"round": rnd + 1, "lr": lr, "train_loss": train_loss, "comm_MB": round(self.comm_mb, 2),
                   "clients": "-".join(map(str, cids)), "steps": sum(m["steps"] for m in msgs)}
            del msgs
            if (rnd + 1) % cfg.eval_every == 0 or rnd + 1 == cfg.rounds:
                ev = self.evaluate(self.val_loader, personalized=cfg.eval_personalized)
                main = ev["main"]
                row.update({"val_loss": main["loss"], "val_mIoU": main["mIoU"],
                            "val_pixel_acc": main["pixel_acc"]})
                if "personalized" in ev and "global" in ev:
                    row["val_mIoU_personalized"] = ev["personalized"]["mIoU"]
                if main["mIoU"] > self.best_score:
                    self.best_score, self.best_round = main["mIoU"], rnd + 1
                    if cfg.save_checkpoints:
                        self._save_best()
            row["time_s"] = round(time.time() - t0, 1)
            self.history.append(row)
            self._write_history()
            if cfg.save_checkpoints:
                torch.save(self._ckpt_payload(rnd), os.path.join(self.out_dir, "last.pt"))
            msg = f"[{cfg.name}] round {rnd + 1:3d}/{cfg.rounds} lr={lr:.4f} train_loss={train_loss:.4f}"
            if "val_mIoU" in row:
                msg += f" | val loss={row['val_loss']:.4f} mIoU={row['val_mIoU']:.4f} acc={row['val_pixel_acc']:.4f}"
            if "val_mIoU_personalized" in row:
                msg += f" pers_mIoU={row['val_mIoU_personalized']:.4f}"
            self.log(msg + f" ({row['time_s']}s)")

        # ---------------- final test with the best (validation-selected) model
        if cfg.save_checkpoints:
            self._load_best()
        test = self.evaluate(self.test_loader, personalized=cfg.eval_personalized or cfg.algorithm == "local"
                             or self.cfg.bn_mode != "avg")
        results = {
            "name": cfg.name, "config": cfg.to_dict(), "best_round": self.best_round,
            "best_val_mIoU": self.best_score, "test": test["main"],
            "test_personalized": test.get("personalized"), "partition": self.partition_stats,
            "comm_MB": self.comm_mb, "params_M": count_params(self.global_model) / 1e6,
            "train_time_min": (time.time() - t_start) / 60,
        }
        if results["test_personalized"]:
            results["test_personalized"] = {k: v for k, v in results["test_personalized"].items() if k != "clients"}
        with open(os.path.join(self.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        self.log(f"[{cfg.name}] TEST (best round {self.best_round}): mIoU={test['main']['mIoU']:.4f} "
                 f"pixel_acc={test['main']['pixel_acc']:.4f} mean_acc={test['main']['mean_acc']:.4f}"
                 + (f" | personalized mIoU={test['personalized']['mIoU']:.4f}" if "personalized" in test else ""))
        try:
            from . import viz
            viz.plot_history(self.out_dir)
            viz.plot_per_class(results, os.path.join(self.out_dir, "per_class_iou.png"))
            if cfg.algorithm != "local":
                self.work_model.load_state_dict(self.global_eval_state())
            else:
                self.work_model.load_state_dict(self._client_model_state(0))
            viz.plot_predictions(self.work_model, self.test_samples, self.bundle, self.img_size, self.device,
                                 os.path.join(self.out_dir, "predictions.png"), cfg.num_vis)
        except Exception as e:  # plotting must never kill a finished run
            self.log(f"[warn] plotting failed: {e}")
        return results


def run_experiment(cfg: Config) -> Dict:
    return FederatedTrainer(cfg).run()
