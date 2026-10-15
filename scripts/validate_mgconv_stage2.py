"""Validate the canonical stage-2 model without training it."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import MGConv  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


STAGE1 = ROOT / "models" / "yolov8n_leyolo_nano_stage1.yaml"
CANONICAL = ROOT / "models" / "yolov8n_leyolo_nano_mgconv_stage2.yaml"
EXPERIMENTAL_B = ROOT / "models" / "experimental_leyolo_nano_mgconv_b_g2.yaml"


def main() -> None:
    import ultralytics

    if Path(ultralytics.__file__).resolve().parent != (ROOT / "third_party" / "ultralytics").resolve():
        raise RuntimeError("Expected project-local Ultralytics")
    if ultralytics.__version__ != "8.4.153":
        raise RuntimeError("Expected Ultralytics 8.4.153")

    original = yaml.safe_load(STAGE1.read_text(encoding="utf-8"))
    canonical = yaml.safe_load(CANONICAL.read_text(encoding="utf-8"))
    expected = deepcopy(original)
    expected["head"][6] = [-1, 1, "MGConv", [64, 3, 2]]
    expected["head"][9] = [-1, 1, "MGConv", [128, 3, 2]]
    if canonical != expected:
        raise AssertionError("Canonical YAML differs from stage 1 outside neck layers 21/24")

    model = YOLO(str(CANONICAL), task="detect", verbose=False).model.eval()
    experimental = YOLO(str(EXPERIMENTAL_B), task="detect", verbose=False).model.eval()
    if [(key, tuple(value.shape)) for key, value in model.state_dict().items()] != [
        (key, tuple(value.shape)) for key, value in experimental.state_dict().items()
    ]:
        raise AssertionError("Canonical MGConv differs from experimental variant B's parameter structure")

    features: dict[str, list[int]] = {}
    branches: dict[str, dict[str, list[int]]] = {}
    handles = []

    def capture_feature(name: str):
        def hook(_module, _inputs, output):
            features[name] = list(output.shape)

        return hook

    def capture_branch(layer: int, name: str):
        def hook(_module, _inputs, output):
            branches.setdefault(str(layer), {})[name] = list(output.shape)

        return hook

    for name, layer in {"p3": 20, "p4": 23, "p5": 26}.items():
        handles.append(model.model[layer].register_forward_hook(capture_feature(name)))
    for layer, channels, spatial in ((21, 64, 40), (24, 128, 20)):
        block = model.model[layer]
        if not isinstance(block, MGConv) or block.groups != 2 or block.main.conv.groups != 2:
            raise AssertionError(f"layer {layer}: expected canonical MGConv with groups=2")
        if block.main.conv.kernel_size != (3, 3) or block.main.conv.stride != (2, 2):
            raise AssertionError(f"layer {layer}: unexpected main kernel/stride")
        if block.shortcut.kernel_size != (1, 1) or block.shortcut.stride != (2, 2):
            raise AssertionError(f"layer {layer}: unexpected residual projection")
        handles.append(block.main.register_forward_hook(capture_branch(layer, "main")))
        handles.append(block.shortcut.register_forward_hook(capture_branch(layer, "shortcut")))

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    if output is None:
        raise AssertionError("Forward pass returned None")
    if features != {"p3": [1, 64, 80, 80], "p4": [1, 128, 40, 40], "p5": [1, 256, 20, 20]}:
        raise AssertionError(f"Unexpected P3/P4/P5: {features}")
    for layer, channels, spatial in ((21, 64, 40), (24, 128, 20)):
        expected_shape = [1, channels, spatial, spatial]
        if branches[str(layer)] != {"main": expected_shape, "shortcut": expected_shape}:
            raise AssertionError(f"Layer {layer} residual mismatch: {branches[str(layer)]}")
    strides = [int(value) for value in model.stride.tolist()]
    if strides != [8, 16, 32] or model.model[-1].nc != 4 or type(model.model[-1]).__name__ != "Detect":
        raise AssertionError("Expected strides 8/16/32, four classes, conventional Detect")
    data_names = list(yaml.safe_load((ROOT / "data" / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    if data_names != ["D00", "D10", "D20", "D40"]:
        raise AssertionError(f"Unexpected dataset class names: {data_names}")

    params = get_num_params(model)
    gflops = round(get_flops(model, imgsz=640), 6)
    if params != get_num_params(experimental) or gflops != round(get_flops(experimental, imgsz=640), 6):
        raise AssertionError("Canonical stage 2 does not match experimental B measurements")
    result = {
        "yaml": str(CANONICAL.relative_to(ROOT)),
        "parameters": params,
        "gflops_640": gflops,
        "delta_parameters_vs_stage1": params - 2_337_500,
        "delta_gflops_640_vs_stage1": round(gflops - 6.257459, 6),
        "classes": data_names,
        "detect_module": type(model.model[-1]).__name__,
        "strides": strides,
        "features": features,
        "residual_branch_shapes": branches,
        "forward_1x3x640x640": "PASS",
        "matches_experimental_B": True,
    }
    output_dir = ROOT / "outputs" / "stage2_mgconv"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structural_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
