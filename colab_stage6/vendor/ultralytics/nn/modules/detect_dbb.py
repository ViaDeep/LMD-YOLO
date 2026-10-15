"""Canonical Detect_DBB head for *our reproduction* of LMD_YOLO.

The paper does not specify six DBBs or their exact positions. This class
canonizes experimental interpretation A: the first 3x3 Conv in cv2 and cv3
at each of P3/P4/P5 is replaced by a reparameterizable DBB.
"""

from __future__ import annotations

from .detect_dbb_experimental import DetectDBB6


class DetectDBB(DetectDBB6):
    """Preserve YOLOv8's decoupled Detect behavior; use six first-position DBBs."""
