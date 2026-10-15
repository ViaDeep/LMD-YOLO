"""LeYOLO backbone blocks used by the LMD_YOLO stage-1 reproduction.

This is a small, auditable port of ``mn_conv`` and ``MobileNetV3_BLOCK``
from Lilian Hollard et al.'s public LeYOLO implementation.  Names are made
explicit to avoid confusing these blocks with Ultralytics' unrelated mobile
modules.

Source: https://github.com/LilianHollard/LeYOLO
Paper: https://arxiv.org/abs/2406.14239

Important fidelity notes:
    * Conv2d bias remains enabled, matching the public LeYOLO code.
    * ``attention`` is accepted for YAML compatibility but is intentionally
      unused, also matching the public implementation.
    * Pointwise expansion is controlled by the explicit ``pointwise`` flag;
      it is not inferred from equal channel counts.
"""

from __future__ import annotations

import torch
from torch import nn

from .conv import autopad


def leyolo_activation(name: str = "RE") -> nn.Module:
    """Return the activation mapping used by the official LeYOLO code."""
    if name == "RE":
        return nn.ReLU6(inplace=True)
    if name == "GE":
        return nn.GELU()
    if name == "SI":
        return nn.SiLU()
    if name == "EL":
        return nn.ELU()
    return nn.Hardswish()


class LeYOLOConv(nn.Module):
    """Faithful port of LeYOLO's ``mn_conv`` (Conv2d + BN + activation)."""

    def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k == s else autopad(k, p, d)
        # bias=True is intentional: it is the default used by the source code.
        self.conv = nn.Conv2d(c1, c2, k, s, padding, dilation=d, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = leyolo_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class LeYOLOInvertedBottleneck(nn.Module):
    """LeYOLO inverted bottleneck from the authors' ``MobileNetV3_BLOCK``."""

    def __init__(
        self,
        c1,
        c2,
        k=3,
        expansion=None,
        attention=False,
        act="RE",
        stride=1,
        pointwise=True,
    ):
        super().__init__()
        del attention  # Present but unused in the official LeYOLO implementation.
        c_mid = c1 if expansion is None else expansion
        self.residual = c1 == c2 and stride == 1

        layers = [LeYOLOConv(c1, c_mid, act=act)] if pointwise else []
        layers.extend(
            [
                LeYOLOConv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                nn.Conv2d(c_mid, c2, 1),
                nn.BatchNorm2d(c2),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return x + y if self.residual else y

