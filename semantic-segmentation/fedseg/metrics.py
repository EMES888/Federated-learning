"""Segmentation metrics from a confusion matrix (void / ignore pixels excluded)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255, device="cpu"):
        self.n = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(1).flatten()
        target = target.flatten()
        keep = (target != self.ignore_index) & (target < self.n)
        idx = target[keep] * self.n + pred[keep]
        self.mat += torch.bincount(idx, minlength=self.n * self.n).view(self.n, self.n)

    def compute(self, class_names: Optional[List[str]] = None) -> Dict:
        m = self.mat.double().cpu().numpy()
        tp = np.diag(m)
        gt, pr = m.sum(1), m.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (gt + pr - tp)
            acc = tp / gt
            dice = 2 * tp / (gt + pr)
        valid = gt > 0   # classes absent from the ground truth are excluded from the means
        out = {
            "mIoU": float(np.nanmean(iou[valid])) if valid.any() else 0.0,
            "pixel_acc": float(tp.sum() / max(m.sum(), 1)),
            "mean_acc": float(np.nanmean(acc[valid])) if valid.any() else 0.0,
            "mDice": float(np.nanmean(dice[valid])) if valid.any() else 0.0,
            # frequency-weighted IoU
            "fwIoU": float(np.nansum((gt / max(gt.sum(), 1)) * np.nan_to_num(iou))),
        }
        names = class_names or [str(i) for i in range(self.n)]
        out["per_class_iou"] = {names[i]: (float(iou[i]) if valid[i] else None) for i in range(self.n)}
        return out
