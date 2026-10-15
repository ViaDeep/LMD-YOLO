"""Structurally compare two non-canonical stage-2 mg_conv hypotheses.

No training, checkpoints, or canonical model changes are performed.
"""

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
from ultralytics.nn.modules import ExperimentalMGConv  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


STAGE1 = ROOT / "models" / "yolov8n_leyolo_nano_stage1.yaml"
VARIANTS = {
    "A_g1": (ROOT / "models" / "experimental_leyolo_nano_mgconv_a_g1.yaml", 1),
    "B_g2": (ROOT / "models" / "experimental_leyolo_nano_mgconv_b_g2.yaml", 2),
}
FEATURES = {"p3": 20, "p4": 23, "p5": 26}
EXPECTED_FEATURES = {
    "p3": [1, 64, 80, 80],
    "p4": [1, 128, 40, 40],
    "p5": [1, 256, 20, 20],
}


def check_yaml_isolation(path: Path, groups: int) -> None:
    """Require exactly two intentional neck-row changes versus stage 1."""
    original = yaml.safe_load(STAGE1.read_text(encoding="utf-8"))
    experimental = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = deepcopy(original)
    for index, channels in ((6, 64), (9, 128)):
        expected["head"][index] = [-1, 1, "ExperimentalMGConv", [channels, 3, 2, groups]]
    if experimental != expected:
        raise AssertionError(f"{path.name}: unexpected changes beyond neck layers 21 and 24")


def measure(path: Path, groups: int) -> dict:
    """Build, forward, and profile one complete candidate."""
    check_yaml_isolation(path, groups)
    wrapper = YOLO(str(path), task="detect", verbose=False)
    model = wrapper.model.eval()
    shapes: dict[str, list[int]] = {}
    branch_shapes: dict[str, dict[str, list[int]]] = {}
    handles = []

    def capture_feature(name: str):
        def hook(_module, _inputs, output):
            shapes[name] = list(output.shape)

        return hook

    def capture_branch(layer: int, branch: str):
        def hook(_module, _inputs, output):
            branch_shapes.setdefault(str(layer), {})[branch] = list(output.shape)

        return hook

    for name, index in FEATURES.items():
        handles.append(model.model[index].register_forward_hook(capture_feature(name)))
    for index, channels in ((21, 64), (24, 128)):
        block = model.model[index]
        if not isinstance(block, ExperimentalMGConv):
            raise AssertionError(f"layer {index}: expected ExperimentalMGConv")
        if block.groups != groups or block.main.conv.groups != groups:
            raise AssertionError(f"layer {index}: incorrect main-path groups")
        if block.main.conv.in_channels != channels or block.main.conv.out_channels != channels:
            raise AssertionError(f"layer {index}: incorrect channels")
        if block.main.conv.kernel_size != (3, 3) or block.main.conv.stride != (2, 2):
            raise AssertionError(f"layer {index}: main-path kernel/stride changed")
        if block.shortcut.kernel_size != (1, 1) or block.shortcut.stride != (2, 2):
            raise AssertionError(f"layer {index}: shortcut kernel/stride changed")
        handles.append(block.main.register_forward_hook(capture_branch(index, "main")))
        handles.append(block.shortcut.register_forward_hook(capture_branch(index, "shortcut")))

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    if output is None:
        raise AssertionError("forward returned None")
    if shapes != EXPECTED_FEATURES:
        raise AssertionError(f"unexpected P3/P4/P5 feature shapes: {shapes}")
    for index in (21, 24):
        branches = branch_shapes[str(index)]
        if branches["main"] != branches["shortcut"]:
            raise AssertionError(f"layer {index}: residual branches do not align: {branches}")

    strides = [int(value) for value in model.stride.tolist()]
    if strides != [8, 16, 32]:
        raise AssertionError(f"incorrect strides: {strides}")
    data = yaml.safe_load((ROOT / "data" / "data.yaml").read_text(encoding="utf-8"))
    names = list(data["names"].values())
    if model.model[-1].nc != 4 or names != ["D00", "D10", "D20", "D40"]:
        raise AssertionError("model/data must define D00, D10, D20, D40")

    return {
        "yaml": str(path.relative_to(ROOT)),
        "groups_main": groups,
        "groups_shortcut": 1,
        "mgconv_layers": [21, 24],
        "parameters": get_num_params(model),
        "gflops_640": round(get_flops(model, imgsz=640), 6),
        "forward_1x3x640x640": "PASS",
        "classes": names,
        "strides": strides,
        "features": shapes,
        "residual_branch_shapes": branch_shapes,
    }


def main() -> None:
    import ultralytics

    if Path(ultralytics.__file__).resolve().parent != (ROOT / "third_party" / "ultralytics").resolve():
        raise RuntimeError("The vendored Ultralytics package must be imported")
    if ultralytics.__version__ != "8.4.153":
        raise RuntimeError("Expected Ultralytics 8.4.153")

    baseline_model = YOLO(str(STAGE1), task="detect", verbose=False).model
    baseline = {
        "parameters": get_num_params(baseline_model),
        "gflops_640": round(get_flops(baseline_model, imgsz=640), 6),
    }
    if baseline != {"parameters": 2_337_500, "gflops_640": 6.257459}:
        raise AssertionError(f"stage-1 structural reference changed unexpectedly: {baseline}")
    results = {}
    for name, (path, groups) in VARIANTS.items():
        result = measure(path, groups)
        result["delta_parameters"] = result["parameters"] - baseline["parameters"]
        result["delta_gflops_640"] = round(result["gflops_640"] - baseline["gflops_640"], 6)
        results[name] = result
    payload = {
        "stage1_reference": baseline,
        "paper_table5_reference": {
            "stage1_parameters_m": 1.39,
            "stage1_gflops_640": 3.9,
            "mgconv_parameters_m": 1.34,
            "mgconv_gflops_640": 3.8,
            "note": "A prose paragraph says 1.36 M; Tables 4 and 5 say 1.34 M.",
        },
        "variants": results,
        "canonical_selection": None,
    }
    output_dir = ROOT / "outputs" / "stage2_mgconv_experimental"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
