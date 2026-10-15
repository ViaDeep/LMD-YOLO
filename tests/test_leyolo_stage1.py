"""Dependency-free structural tests for the two stage-1 LeYOLO candidates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import LeYOLOConv, LeYOLOInvertedBottleneck  # noqa: E402


class TestLeYOLOStage1(unittest.TestCase):
    def check_candidate(self, yaml_name: str, channels: list[int]) -> None:
        wrapper = YOLO(str(ROOT / "models" / yaml_name), task="detect", verbose=False)
        model = wrapper.model.eval()
        observed: dict[str, list[int]] = {}

        def capture(name: str):
            def hook(_module, _inputs, output):
                observed[name] = list(output.shape)

            return hook

        handles = []
        for key, index in {"p3": 4, "p4": 9, "p5": 14}.items():
            handles.append(model.model[index].register_forward_hook(capture(key)))
        with torch.inference_mode():
            model(torch.zeros(1, 3, 640, 640))
        for handle in handles:
            handle.remove()

        self.assertEqual([int(x) for x in model.stride.tolist()], [8, 16, 32])
        self.assertEqual(model.model[-1].nc, 4)
        self.assertEqual(
            observed,
            {
                "p3": [1, channels[0], 80, 80],
                "p4": [1, channels[1], 40, 40],
                "p5": [1, channels[2], 20, 20],
            },
        )

    def test_nano_shapes_and_strides(self) -> None:
        self.check_candidate("yolov8n_leyolo_nano_stage1.yaml", [32, 64, 96])

    def test_small_shapes_and_strides(self) -> None:
        self.check_candidate("yolov8n_leyolo_small_stage1.yaml", [48, 88, 128])

    def test_official_pointwise_and_residual_semantics(self) -> None:
        no_pointwise = LeYOLOInvertedBottleneck(16, 16, 3, 16, False, "SI", 2, False)
        residual = LeYOLOInvertedBottleneck(32, 32, 3, 96, False, "SI", 1, True)
        self.assertEqual(sum(isinstance(module, LeYOLOConv) for module in no_pointwise.layers), 1)
        self.assertTrue(residual.residual)


if __name__ == "__main__":
    unittest.main()
