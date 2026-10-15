"""Dependency-free focused checks for the experimental mg_conv variants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics.nn.modules import ExperimentalMGConv  # noqa: E402


class TestExperimentalMGConv(unittest.TestCase):
    def test_branch_shapes_and_activation(self) -> None:
        for channels, size, groups in ((64, 80, 1), (128, 40, 2)):
            with self.subTest(channels=channels, groups=groups):
                block = ExperimentalMGConv(channels, channels, 3, 2, groups).eval()
                self.assertIsInstance(block.main.act, nn.GELU)
                self.assertIsInstance(block.shortcut, nn.Conv2d)
                self.assertEqual(block.main.conv.groups, groups)
                with torch.inference_mode():
                    x = torch.zeros(1, channels, size, size)
                    self.assertEqual(block.main(x).shape, block.shortcut(x).shape)
                    self.assertEqual(list(block(x).shape), [1, channels, size // 2, size // 2])

    def test_invalid_groups_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentalMGConv(64, 128, 3, 2, 3)


if __name__ == "__main__":
    unittest.main()
