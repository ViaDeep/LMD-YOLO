"""Structural and numerical TRAIN/DEPLOY checks for two noncanonical DBB heads."""

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
from ultralytics.nn.modules import DiverseBranchBlock  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


STAGE2 = ROOT / "models" / "yolov8n_leyolo_nano_mgconv_stage2.yaml"
VARIANTS = {
    "A": (ROOT / "models" / "experimental_leyolo_nano_mgconv_detect_dbb_a6.yaml", 6),
    "B": (ROOT / "models" / "experimental_leyolo_nano_mgconv_detect_dbb_b12.yaml", 12),
}


def errors(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = (left - right).abs()
    return {
        "max_abs": float(delta.max()),
        "max_rel": float((delta / left.abs().clamp_min(1e-6)).max()),
        "shape": list(left.shape),
    }


def compare(left: torch.Tensor, right: torch.Tensor, rtol: float, atol: float) -> dict:
    result = errors(left, right)
    torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    return result


def captures(model, x: torch.Tensor):
    head = model.model[-1]
    raw = {}
    features = {}
    handles = []
    for scale, layer in enumerate((20, 23, 26)):
        handles.append(model.model[layer].register_forward_hook(
            lambda _m, _i, out, scale=scale: features.__setitem__(f"P{scale+3}", list(out.shape))
        ))
        for tower_name in ("cv2", "cv3"):
            tower = getattr(head, tower_name)[scale]
            handles.append(tower[-1].register_forward_hook(
                lambda _m, _i, out, scale=scale, name=tower_name: raw.__setitem__(f"P{scale+3}_{name}", out.detach().clone())
            ))
    try:
        with torch.no_grad():
            decoded, predictions = model(x)
    finally:
        for handle in handles:
            handle.remove()
    return decoded, predictions, raw, features


def run_variant(name: str, yaml_path: Path, count: int, x: torch.Tensor) -> dict:
    expected = yaml.safe_load(STAGE2.read_text(encoding="utf-8"))
    actual = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    expected["head"][-1][2] = f"DetectDBB{count}"
    if actual != expected:
        raise AssertionError(f"Variant {name} differs from frozen stage 2 beyond the Detect class")

    model = YOLO(str(yaml_path), task="detect", verbose=False).model.eval()
    blocks = [m for m in model.modules() if isinstance(m, DiverseBranchBlock)]
    if len(blocks) != count or model.model[-1].nc != 4:
        raise AssertionError(f"Variant {name}: incorrect DBB count or class count")
    if [int(v) for v in model.stride.tolist()] != [8, 16, 32]:
        raise AssertionError(f"Variant {name}: incorrect strides")
    model.train()
    with torch.no_grad():
        train_output = model(x)
    if train_output["boxes"].shape != (1, 64, 8400) or train_output["scores"].shape != (1, 4, 8400):
        raise AssertionError(f"Variant {name}: training-mode forward returned incorrect logits")
    model.eval()

    per_block = []
    for index, block in enumerate(blocks):
        isolated = deepcopy(block).eval()
        # Non-default running statistics expose incorrect BN-aware border padding.
        with torch.no_grad():
            for bn in (m for m in isolated.modules() if isinstance(m, torch.nn.BatchNorm2d)):
                bn.running_mean.uniform_(-0.2, 0.2)
                bn.running_var.uniform_(0.5, 1.5)
                bn.weight.uniform_(0.7, 1.3)
                bn.bias.uniform_(-0.2, 0.2)
        probe = torch.randn(1, block.in_channels, 11, 13)
        with torch.no_grad():
            before = isolated(probe)
        converted = deepcopy(isolated).eval()
        converted.switch_to_deploy()
        with torch.no_grad():
            after = converted(probe)
        discrepancy = compare(before, after, rtol=1e-4, atol=1e-5)
        edge = torch.cat((
            (before - after).abs()[:, :, 0, :].flatten(),
            (before - after).abs()[:, :, -1, :].flatten(),
            (before - after).abs()[:, :, :, 0].flatten(),
            (before - after).abs()[:, :, :, -1].flatten(),
        ))
        discrepancy["max_abs_border"] = float(edge.max())
        discrepancy["block"] = index
        per_block.append(discrepancy)

    train_params = get_num_params(model)
    train_gflops = get_flops(model, imgsz=640)
    before_decoded, before_pred, before_raw, before_features = captures(model, x)

    deployed = deepcopy(model).eval()
    for block in deployed.modules():
        if isinstance(block, DiverseBranchBlock):
            block.switch_to_deploy()
    deployed_blocks = [m for m in deployed.modules() if isinstance(m, DiverseBranchBlock)]
    if len(deployed_blocks) != count or any(
        not hasattr(m, "dbb_reparam") or any(hasattr(m, attr) for attr in ("dbb_origin", "dbb_1x1", "dbb_avg", "dbb_1x1_kxk"))
        for m in deployed_blocks
    ):
        raise AssertionError(f"Variant {name}: deploy model still has training branches")
    if any(hasattr(m, "dbb_reparam") for m in blocks):
        raise AssertionError(f"Variant {name}: conversion mutated the training model")

    deploy_params = get_num_params(deployed)
    deploy_gflops = get_flops(deployed, imgsz=640)
    after_decoded, after_pred, after_raw, after_features = captures(deployed, x)
    expected_features = {"P3": [1, 64, 80, 80], "P4": [1, 128, 40, 40], "P5": [1, 256, 20, 20]}
    if before_features != expected_features or after_features != expected_features:
        raise AssertionError(f"Variant {name}: incorrect P3/P4/P5 features")
    if before_pred["boxes"].shape != (1, 64, 8400) or before_pred["scores"].shape != (1, 4, 8400):
        raise AssertionError(f"Variant {name}: incorrect Detect logits")

    full = {key: compare(before_raw[key], after_raw[key], 1e-3, 1e-4) for key in before_raw}
    full["boxes_concat"] = compare(before_pred["boxes"], after_pred["boxes"], 1e-3, 1e-4)
    full["scores_concat"] = compare(before_pred["scores"], after_pred["scores"], 1e-3, 1e-4)
    full["decoded"] = compare(before_decoded, after_decoded, 1e-3, 1e-4)
    if not train_gflops or not deploy_gflops:
        raise AssertionError(f"Variant {name}: THOP profiling failed")
    return {
        "yaml": str(yaml_path.relative_to(ROOT)),
        "head": type(model.model[-1]).__name__,
        "dbb_count": count,
        "classes": 4,
        "strides": [8, 16, 32],
        "features_train": before_features,
        "features_deploy": after_features,
        "train": {"parameters": train_params, "gflops_640": round(train_gflops, 6)},
        "deploy": {"parameters": deploy_params, "gflops_640": round(deploy_gflops, 6)},
        "delta_vs_stage2_train": {"parameters": train_params - 2_266_204, "gflops_640": round(train_gflops - 6.165709, 6)},
        "delta_vs_stage2_deploy": {"parameters": deploy_params - 2_266_204, "gflops_640": round(deploy_gflops - 6.165709, 6)},
        "block_equivalence": {
            "rtol": 1e-4, "atol": 1e-5,
            "max_abs": max(item["max_abs"] for item in per_block),
            "max_rel": max(item["max_rel"] for item in per_block),
            "max_abs_border": max(item["max_abs_border"] for item in per_block),
            "details": per_block,
        },
        "full_equivalence": {"rtol": 1e-3, "atol": 1e-4, "details": full},
        "training_branches_removed_in_deploy": True,
        "training_model_preserved": True,
        "forward_1x3x640x640": "PASS",
        "training_mode_forward_1x3x640x640": "PASS",
    }


def main() -> None:
    import ultralytics

    if ultralytics.__version__ != "8.4.153" or Path(ultralytics.__file__).resolve().parent != (ROOT / "third_party" / "ultralytics").resolve():
        raise RuntimeError("Expected project-local Ultralytics 8.4.153")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    names = list(yaml.safe_load((ROOT / "data" / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    if names != ["D00", "D10", "D20", "D40"]:
        raise AssertionError("Unexpected dataset classes")
    x = torch.randn(1, 3, 640, 640)
    result = {"stage2_reference": {"parameters": 2_266_204, "gflops_640": 6.165709}, "classes": names}
    for name, (path, count) in VARIANTS.items():
        result[name] = run_variant(name, path, count, x)
        print(f"{name}: TRAIN {result[name]['train']}; DEPLOY {result[name]['deploy']}", flush=True)
    output = ROOT / "outputs" / "detect_dbb_experimental"
    output.mkdir(parents=True, exist_ok=True)
    (output / "structural_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("A", "B")}, indent=2))


if __name__ == "__main__":
    main()
