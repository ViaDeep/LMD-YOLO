"""Project-local adaptation of Ding et al.'s DiverseBranchBlock (Apache-2.0).

Source: https://github.com/DingXiaoH/DiverseBranchBlock/blob/master/diversebranchblock.py
Only the dense-convolution case used by the experimental Detect heads is supported.
The branch ordering, BN-aware padding, and kernel fusion follow the official code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BNAndPadLayer(nn.Module):
    """Pad with the BN transform of zero, as required for exact border fusion."""

    def __init__(self, channels: int, padding: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        if self.padding:
            bn = self.bn
            value = bn.bias.detach() - bn.running_mean * bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
            out = F.pad(out, [self.padding] * 4)
            value = value.reshape(1, -1, 1, 1)
            p = self.padding
            out[:, :, :p, :] = value
            out[:, :, -p:, :] = value
            out[:, :, :, :p] = value
            out[:, :, :, -p:] = value
        return out


class IdentityBasedConv1x1(nn.Conv2d):
    """Official identity-initialized 1x1 branch, including the trainable offset."""

    def __init__(self, channels: int):
        super().__init__(channels, channels, 1, bias=False)
        self.register_buffer("identity", torch.eye(channels).reshape(channels, channels, 1, 1), persistent=False)
        nn.init.zeros_(self.weight)

    def actual_kernel(self) -> torch.Tensor:
        return self.weight + self.identity.to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.actual_kernel())


def _conv_bn(c1: int, c2: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c1, c2, kernel, stride, padding, bias=False), nn.BatchNorm2d(c2))


def _fuse_bn(kernel: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return kernel * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale


def _compose(k1: torch.Tensor, b1: torch.Tensor, k2: torch.Tensor, b2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Official transIII_1x1_kxk, groups=1.
    kernel = F.conv2d(k2, k1.permute(1, 0, 2, 3))
    bias = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
    return kernel, bias


class DiverseBranchBlock(nn.Module):
    """Official four-branch DBB specialization: odd kernel, dense groups=1."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3,
        stride: int = 1, padding: int = 1, groups: int = 1,
        deploy: bool = False, nonlinear: nn.Module | None = None,
    ):
        super().__init__()
        if groups != 1 or kernel_size != 3 or stride != 1 or padding != 1:
            raise ValueError("Experimental Detect DBB supports only kernel=3, stride=1, padding=1, groups=1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()
        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
            return

        self.dbb_origin = _conv_bn(in_channels, out_channels, 3, 1, 1)
        self.dbb_1x1 = _conv_bn(in_channels, out_channels, 1, 1, 0)
        self.dbb_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BNAndPadLayer(out_channels, 1),
            nn.AvgPool2d(3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.dbb_1x1_kxk = nn.Sequential(
            IdentityBasedConv1x1(in_channels),
            BNAndPadLayer(in_channels, 1),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "dbb_reparam"):
            return self.nonlinear(self.dbb_reparam(x))
        out = self.dbb_origin(x) + self.dbb_1x1(x)
        out = out + self.dbb_avg(x) + self.dbb_1x1_kxk(x)
        return self.nonlinear(out)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror official BN fusion, sequential composition, average kernel and sum."""
        if hasattr(self, "dbb_reparam"):
            return self.dbb_reparam.weight, self.dbb_reparam.bias
        origin_k, origin_b = _fuse_bn(self.dbb_origin[0].weight, self.dbb_origin[1])
        one_k, one_b = _fuse_bn(self.dbb_1x1[0].weight, self.dbb_1x1[1])
        one_k = F.pad(one_k, [1, 1, 1, 1])

        seq_k1, seq_b1 = _fuse_bn(self.dbb_1x1_kxk[0].actual_kernel(), self.dbb_1x1_kxk[1].bn)
        seq_k2, seq_b2 = _fuse_bn(self.dbb_1x1_kxk[2].weight, self.dbb_1x1_kxk[3])
        seq_k, seq_b = _compose(seq_k1, seq_b1, seq_k2, seq_b2)

        avg_kernel = torch.zeros((self.out_channels, self.out_channels, 3, 3), device=origin_k.device, dtype=origin_k.dtype)
        indices = torch.arange(self.out_channels, device=origin_k.device)
        avg_kernel[indices, indices] = 1.0 / 9.0
        avg_k2, avg_b2 = _fuse_bn(avg_kernel, self.dbb_avg[3])
        avg_k1, avg_b1 = _fuse_bn(self.dbb_avg[0].weight, self.dbb_avg[1].bn)
        avg_k, avg_b = _compose(avg_k1, avg_b1, avg_k2, avg_b2)
        return origin_k + one_k + seq_k + avg_k, origin_b + one_b + seq_b + avg_b

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if hasattr(self, "dbb_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=True)
        conv = conv.to(device=kernel.device, dtype=kernel.dtype)
        conv.weight.copy_(kernel)
        conv.bias.copy_(bias)
        self.dbb_reparam = conv
        del self.dbb_origin, self.dbb_1x1, self.dbb_avg, self.dbb_1x1_kxk
