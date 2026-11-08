"""Segmentation networks. All models accept any input size (padded internally to a multiple of 32)."""
from __future__ import annotations

from typing import Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(min(32, ch // 4) if ch >= 8 else 1, ch)
    raise ValueError(kind)


class SegNetBasic(nn.Module):
    """SegNet-Basic (Badrinarayanan et al., 2017): 4 encoders / 4 decoders, 64 ch, 7x7 kernels,
    max-pool indices reused for upsampling.
    (As in the paper, decoder convolutions have BN but no ReLU.)"""

    def __init__(self, in_ch: int, num_classes: int, width: int = 64, k: int = 7):
        super().__init__()
        p = k // 2
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch = in_ch
        for _ in range(4):
            self.enc.append(nn.Sequential(nn.Conv2d(ch, width, k, padding=p, bias=False),
                                          nn.BatchNorm2d(width), nn.ReLU(inplace=True)))
            ch = width
        for _ in range(4):
            self.dec.append(nn.Sequential(nn.Conv2d(width, width, k, padding=p, bias=False),
                                          nn.BatchNorm2d(width)))
        self.classifier = nn.Conv2d(width, num_classes, 1)

    def forward(self, x):
        indices, sizes = [], []
        for enc in self.enc:
            x = enc(x)
            sizes.append(x.shape[-2:])
            x, idx = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(idx)
        for dec, idx, size in zip(self.dec, reversed(indices), reversed(sizes)):
            x = F.max_unpool2d(x, idx, 2, 2, output_size=size)
            x = dec(x)
        return self.classifier(x)


class _DoubleConv(nn.Sequential):
    def __init__(self, cin, cout, norm):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), _norm(norm, cout),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), _norm(norm, cout),
                         nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Compact U-Net (Ronneberger et al., 2015) with BatchNorm or GroupNorm."""

    def __init__(self, in_ch: int, num_classes: int, base: int = 32, norm: str = "bn"):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList([_DoubleConv(in_ch, c[0], norm)] +
                                   [_DoubleConv(c[i], c[i + 1], norm) for i in range(4)])
        self.ups = nn.ModuleList([_DoubleConv(c[i + 1] + c[i], c[i], norm) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for up in self.ups:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, s], 1))
        return self.head(x)


class _TorchvisionSeg(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]


class PadWrapper(nn.Module):
    """Pads the input to a multiple of `multiple` and crops the logits back."""

    def __init__(self, net: nn.Module, multiple: int = 32):
        super().__init__()
        self.net, self.multiple = net, multiple

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.multiple, (-w) % self.multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if min(h, w) > max(ph, pw) else "constant")
        out = self.net(x)
        return out[..., :h, :w]


def build_model(name: str, num_classes: int, pretrained: bool = True, norm: str = "bn",
                unet_base: int = 32) -> nn.Module:
    if name == "segnet":
        net = SegNetBasic(3, num_classes)
    elif name == "unet":
        net = UNet(3, num_classes, base=unet_base, norm=norm)
    elif name in ("deeplabv3_mbv3", "lraspp_mbv3"):
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.segmentation import (deeplabv3_mobilenet_v3_large,
                                                     lraspp_mobilenet_v3_large)
        fn = deeplabv3_mobilenet_v3_large if name == "deeplabv3_mbv3" else lraspp_mobilenet_v3_large
        wb = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        kwargs = {"aux_loss": False} if name == "deeplabv3_mbv3" else {}
        net = _TorchvisionSeg(fn(weights=None, weights_backbone=wb, num_classes=num_classes, **kwargs))
    else:
        raise ValueError(f"unknown model {name!r}")
    return PadWrapper(net)


def norm_layer_names(model: nn.Module) -> Tuple[Set[str], Set[str]]:
    """Returns (param names, buffer names) that belong to BatchNorm layers (for FedBN / SiloBN)."""
    params, buffers = set(), set()
    for mname, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            prefix = mname + "." if mname else ""
            params |= {prefix + n for n, _ in m.named_parameters(recurse=False)}
            buffers |= {prefix + n for n, _ in m.named_buffers(recurse=False)}
    return params, buffers


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
