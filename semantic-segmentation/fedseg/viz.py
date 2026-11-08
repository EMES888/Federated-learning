"""Plots for single runs and run comparisons (matplotlib, Agg backend - safe on Colab/headless)."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

COLORS = ["#2a6fdb", "#e5732e", "#2e9e5b", "#c23b5a", "#7b5cc4", "#8a6d3b", "#d14fa8", "#5f6b7a",
          "#18a2b8", "#b5a300"]


def plot_history(run_dir: str) -> None:
    path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(path):
        return
    h = pd.read_csv(path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["round"], h["train_loss"], label="train (client avg)", color=COLORS[0])
    if "val_loss" in h:
        ax[0].plot(h["round"], h["val_loss"], label="validation", color=COLORS[1])
    ax[0].set(xlabel="communication round", ylabel="cross-entropy", title="Loss")
    ax[0].legend(frameon=False)
    if "val_mIoU" in h:
        ax[1].plot(h["round"], h["val_mIoU"], label="global", color=COLORS[2])
    if "val_mIoU_personalized" in h:
        ax[1].plot(h["round"], h["val_mIoU_personalized"], label="personalised", color=COLORS[3])
    ax[1].set(xlabel="communication round", ylabel="mIoU", title="Validation mIoU")
    ax[1].legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "curves.png"), dpi=130)
    plt.close(fig)


def plot_per_class(results: Dict, path: str) -> None:
    pc = results["test"]["per_class_iou"]
    names = [k for k, v in pc.items() if v is not None]
    vals = [pc[k] for k in names]
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(names)), 3.6))
    ax.bar(names, vals, color=COLORS[0])
    ax.set(ylim=(0, 1), ylabel="IoU", title=f"{results['name']}  test mIoU={results['test']['mIoU']:.3f}")
    ax.tick_params(axis="x", rotation=60)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    valid = mask < len(palette)
    out[valid] = palette[mask[valid]]
    return out


@torch.no_grad()
def plot_predictions(model, samples, bundle, img_size, device, path: str, n: int = 6) -> None:
    from .data import SegDataset
    from .data.common import IMAGENET_MEAN, IMAGENET_STD
    if n <= 0 or not samples:
        return
    idx = np.linspace(0, len(samples) - 1, min(n, len(samples))).astype(int)
    ds = SegDataset([samples[i] for i in idx], bundle.load_mask, img_size, train=False)
    model.eval()
    fig, axes = plt.subplots(len(ds), 3, figsize=(10, 2.8 * len(ds)), squeeze=False)
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    for r in range(len(ds)):
        x, y = ds[r]
        pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
        img = (x.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        for c, (im, title) in enumerate([(img, "image"), (colorize(y.numpy(), bundle.palette), "ground truth"),
                                         (colorize(pred, bundle.palette), "prediction")]):
            axes[r, c].imshow(im)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- comparisons
def load_runs(root: str, pattern: str = "*") -> List[Dict]:
    runs = []
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        rp = os.path.join(d, "results.json")
        if os.path.exists(rp):
            with open(rp) as f:
                r = json.load(f)
            hp = os.path.join(d, "history.csv")
            r["history"] = pd.read_csv(hp) if os.path.exists(hp) else None
            r["dir"] = d
            runs.append(r)
    return runs


def summary_table(runs: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        c = r["config"]
        pers = r.get("test_personalized") or {}
        rows.append({
            "run": r["name"], "dataset": c["dataset"], "model": c["model"], "algorithm": c["algorithm"],
            "bn_mode": c["bn_mode"], "server_opt": c["server_opt"], "partition": c["partition"],
            "clients": c["num_clients"], "seed": c["seed"],
            "test_mIoU": r["test"]["mIoU"], "test_pixel_acc": r["test"]["pixel_acc"],
            "test_mean_acc": r["test"]["mean_acc"], "pers_mIoU": pers.get("mIoU"),
            "best_round": r["best_round"], "comm_MB": r["comm_MB"],
            "time_min": r.get("train_time_min"), "JS_skew": r["partition"].get("mean_js"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "test_mIoU"], ascending=[True, False]) if len(df) else df


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """mean +- std over seeds for otherwise identical configurations."""
    keys = ["dataset", "model", "algorithm", "bn_mode", "server_opt", "partition", "clients"]
    g = df.groupby(keys)
    out = g["test_mIoU"].agg(["mean", "std", "count"]).reset_index()
    return out.sort_values(["dataset", "mean"], ascending=[True, False])


def plot_comparison(runs: List[Dict], path: str, metric: str = "val_mIoU", title: Optional[str] = None,
                    x: str = "round") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, r in enumerate(runs):
        h = r["history"]
        if h is None or metric not in h:
            continue
        hh = h.dropna(subset=[metric])
        ax.plot(hh[x], hh[metric], label=r["name"], color=COLORS[i % len(COLORS)], lw=1.8)
    ax.set(xlabel="communication round" if x == "round" else x, ylabel=metric, title=title or metric)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
