"""Two *experimental* DBB placements; neither is claimed as the LMD_YOLO head."""

from __future__ import annotations

from torch import nn

from .diverse_branch_block import DiverseBranchBlock
from .head import Detect


class _DetectDBBExperimental(Detect):
    """Keep Detect's decoupled towers, DFL, decoding, and final projections."""

    replace_per_tower: int = 0

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if not self.legacy:
            raise ValueError("Experimental DBB heads require the YOLOv8 legacy (dense) Detect towers")
        for towers in (self.cv2, self.cv3):
            for tower in towers:
                for index in range(self.replace_per_tower):
                    original = tower[index]
                    conv = original.conv
                    if conv.kernel_size != (3, 3) or conv.stride != (1, 1) or conv.padding != (1, 1) or conv.groups != 1:
                        raise ValueError("Expected a dense 3x3/stride-1 Conv in a legacy Detect tower")
                    tower[index] = DiverseBranchBlock(
                        conv.in_channels, conv.out_channels, 3, 1, 1, 1,
                        nonlinear=nn.SiLU(),
                    )


class DetectDBB6(_DetectDBBExperimental):
    """A: replace the first 3x3 Conv in each cv2/cv3 tower (6 DBB)."""

    replace_per_tower = 1


class DetectDBB12(_DetectDBBExperimental):
    """B: replace both 3x3 Convs in each cv2/cv3 tower (12 DBB)."""

    replace_per_tower = 2
