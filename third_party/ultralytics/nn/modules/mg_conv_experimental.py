"""Experimental interpretations of the LMD_YOLO mg_conv diagram.

The paper specifies the main Conv2d-BatchNorm-GELU path and depicts a bare
Conv2d projection on the residual path. It does not specify either path's
kernel, stride, channel count, bias, or group count. The stage-2 comparison
inherits main-path channels/kernel/stride from the two replaced YOLOv8n neck
convolutions, uses a 1x1 stride-matched projection, and compares groups 1/2.
None of those choices is claimed to be the authors' unpublished source code.
"""

from __future__ import annotations

import torch
from torch import nn

from .leyolo import LeYOLOConv


class ExperimentalMGConv(nn.Module):
    """Conv2d-BN-GELU plus a shape-compatible Conv2d residual projection."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, g: int = 1):
        super().__init__()
        if g < 1 or c1 % g or c2 % g:
            raise ValueError(f"groups={g} must divide both input={c1} and output={c2} channels")
        if k < 1 or s < 1:
            raise ValueError(f"kernel={k} and stride={s} must be positive")
        self.groups = g
        # LeYOLOConv is the already-audited mn_conv port. GE maps to nn.GELU.
        # Its Conv2d bias=True matches the public LeYOLO implementation.
        self.main = LeYOLOConv(c1, c2, k=k, s=s, act="GE", g=g)
        # Figure 7 shows a bare Conv2d on the skip branch; 1x1, s and groups=1
        # are explicit comparison assumptions needed to align tensor dimensions.
        self.shortcut = nn.Conv2d(c1, c2, kernel_size=1, stride=s, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main(x)
        shortcut = self.shortcut(x)
        if main.shape != shortcut.shape:
            raise RuntimeError(f"mg_conv branch mismatch: {main.shape} vs {shortcut.shape}")
        return main + shortcut
