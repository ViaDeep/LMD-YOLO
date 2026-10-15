"""Canonical stage-2 mg_conv choice for this LMD_YOLO reproduction.

The paper does NOT publish an effective group count. ``groups=2`` is our
documented reproduction decision, not a claim about the authors' model.
The implementation reuses the structurally validated experimental block so
that making variant B canonical cannot change its numerical architecture.
"""

from __future__ import annotations

from .mg_conv_experimental import ExperimentalMGConv


class MGConv(ExperimentalMGConv):
    """Variant B: grouped 3x3 Conv-BN-GELU plus 1x1 stride-matched projection."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__(c1=c1, c2=c2, k=k, s=s, g=2)
