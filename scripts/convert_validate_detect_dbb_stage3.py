"""Convert a trained stage-3 best.pt on a copy and verify FP32 equivalence."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import DetectDBB, DiverseBranchBlock  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402
from compare_detect_dbb_experimental import compare, errors  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(model, image: torch.Tensor):
    raw: dict[str, torch.Tensor] = {}
    features: dict[str, torch.Tensor] = {}
    dbb_inputs: dict[str, torch.Tensor] = {}
    handles = []
    head = model.model[-1]
    for scale, layer in enumerate((20, 23, 26)):
        name = f"P{scale + 3}"
        handles.append(model.model[layer].register_forward_hook(
            lambda _m, _i, out, name=name: features.__setitem__(name, out.detach().clone())
        ))
        for branch in ("cv2", "cv3"):
            tower = getattr(head, branch)[scale]
            key = f"{name}_{branch}"
            handles.append(tower[0].register_forward_pre_hook(
                lambda _m, args, key=key: dbb_inputs.__setitem__(key, args[0].detach().clone())
            ))
            handles.append(tower[-1].register_forward_hook(
                lambda _m, _i, out, key=key: raw.__setitem__(key, out.detach().clone())
            ))
    try:
        with torch.no_grad():
            decoded, predictions = model(image)
    finally:
        for handle in handles:
            handle.remove()
    return decoded, predictions, raw, features, dbb_inputs


def check_deploy(model) -> None:
    if type(model.model[-1]) is not DetectDBB:
        raise AssertionError("Expected canonical DetectDBB checkpoint")
    blocks = [m for m in model.modules() if isinstance(m, DiverseBranchBlock)]
    if len(blocks) != 6:
        raise AssertionError(f"Expected six DBBs, found {len(blocks)}")
    if any(
        not hasattr(m, "dbb_reparam") or any(hasattr(m, name) for name in ("dbb_origin", "dbb_1x1", "dbb_avg", "dbb_1x1_kxk"))
        for m in blocks
    ):
        raise AssertionError("Deploy checkpoint retains training branches")


def main() -> None:
    import ultralytics

    # Use deterministic native FP32 convolution arithmetic for the strict
    # block-level equivalence check. A separate audit records MKLDNN behavior.
    torch.backends.mkldnn.enabled = False
    if ultralytics.__version__ != "8.4.153" or Path(ultralytics.__file__).resolve().parent != (ROOT / "third_party" / "ultralytics").resolve():
        raise RuntimeError("Expected project-local Ultralytics 8.4.153")
    torch.set_num_threads(4)
    best = ROOT / "runs" / "detect" / "leyolo_nano_mgconv_dbb_stage3_smoke" / "weights" / "best.pt"
    deploy_dir = ROOT / "runs" / "detect" / "leyolo_nano_mgconv_dbb_stage3_deploy" / "weights"
    deployed_path = deploy_dir / "best_deploy_fp32.pt"
    output_path = ROOT / "outputs" / "stage3_detect_dbb" / "deploy_validation.json"
    if not best.is_file():
        raise FileNotFoundError("Stage-3 best.pt does not exist")
    if deployed_path.exists() or output_path.exists():
        raise FileExistsError("Refusing to overwrite an existing deploy checkpoint/report")
    best_hash = sha256(best)
    image_path = next(iter(sorted((ROOT / "data" / "dataset" / "images" / "val").glob("*.jpg"))), None)
    if image_path is None:
        raise FileNotFoundError("No real validation JPG found")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {image_path}")
    rgb = cv2.cvtColor(cv2.resize(bgr, (640, 640)), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)

    loaded = YOLO(str(best), task="detect")
    train_model = loaded.model.float().eval()
    if type(train_model.model[-1]) is not DetectDBB or train_model.model[-1].nc != 4:
        raise AssertionError("best.pt is not canonical stage-3 DetectDBB")
    train_blocks = [m for m in train_model.modules() if isinstance(m, DiverseBranchBlock)]
    if len(train_blocks) != 6 or any(hasattr(m, "dbb_reparam") for m in train_blocks):
        raise AssertionError("best.pt must contain six multibranch DBBs")
    if [int(v) for v in train_model.stride.tolist()] != [8, 16, 32]:
        raise AssertionError("Unexpected best.pt strides")
    before, before_pred, before_raw, before_features, inputs = capture(train_model, image)
    if {k: list(v.shape) for k, v in before_features.items()} != {
        "P3": [1, 64, 80, 80], "P4": [1, 128, 40, 40], "P5": [1, 256, 20, 20]
    }:
        raise AssertionError("Unexpected best.pt P3/P4/P5")

    deployed = deepcopy(train_model).eval()
    for block in deployed.modules():
        if isinstance(block, DiverseBranchBlock):
            block.switch_to_deploy()
    check_deploy(deployed)
    per_block = {}
    for scale in range(3):
        for branch in ("cv2", "cv3"):
            key = f"P{scale + 3}_{branch}"
            original = getattr(train_model.model[-1], branch)[scale][0]
            converted = getattr(deployed.model[-1], branch)[scale][0]
            with torch.no_grad():
                left = original(inputs[key])
                right = converted(inputs[key])
            try:
                discrepancy = compare(left, right, 1e-4, 1e-5)
            except AssertionError as exc:
                raise AssertionError(f"Trained DBB {key} failed: {errors(left, right)}") from exc
            edges = torch.cat((
                (left - right).abs()[:, :, 0, :].flatten(),
                (left - right).abs()[:, :, -1, :].flatten(),
                (left - right).abs()[:, :, :, 0].flatten(),
                (left - right).abs()[:, :, :, -1].flatten(),
            ))
            discrepancy["max_abs_border"] = float(edges.max())
            per_block[key] = discrepancy
    after, after_pred, after_raw, after_features, _ = capture(deployed, image)
    full = {key: compare(before_raw[key], after_raw[key], 1e-3, 1e-4) for key in before_raw}
    for key in before_features:
        full[f"{key}_feature"] = compare(before_features[key], after_features[key], 1e-3, 1e-4)
    full["boxes_concat"] = compare(before_pred["boxes"], after_pred["boxes"], 1e-3, 1e-4)
    full["scores_concat"] = compare(before_pred["scores"], after_pred["scores"], 1e-3, 1e-4)
    full["decoded"] = compare(before, after, 1e-3, 1e-4)
    if any(hasattr(m, "dbb_reparam") for m in train_blocks):
        raise AssertionError("Training model mutated by deploy conversion")

    deploy_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = dict(loaded.ckpt)
    checkpoint.update({"model": deployed, "ema": None, "optimizer": None, "dbb_deploy": True})
    torch.save(checkpoint, deployed_path)
    reloaded = YOLO(str(deployed_path), task="detect").model.float().eval()
    check_deploy(reloaded)
    reloaded_out, reloaded_pred, reloaded_raw, reloaded_features, _ = capture(reloaded, image)
    reload_comparison = {k: compare(after_raw[k], reloaded_raw[k], 1e-3, 1e-4) for k in after_raw}
    reload_comparison["decoded"] = compare(after, reloaded_out, 1e-3, 1e-4)
    if {k: list(v.shape) for k, v in reloaded_features.items()} != {k: list(v.shape) for k, v in after_features.items()}:
        raise AssertionError("Reloaded deploy checkpoint changed feature shapes")
    if before_pred["boxes"].shape != reloaded_pred["boxes"].shape or before_pred["scores"].shape != reloaded_pred["scores"].shape:
        raise AssertionError("Reloaded deploy checkpoint changed output shapes")
    if sha256(best) != best_hash:
        raise AssertionError("Original best.pt changed during conversion")

    result = {
        "train_checkpoint": str(best), "train_checkpoint_sha256_unchanged": best_hash,
        "train_checkpoint_bytes": best.stat().st_size,
        "deploy_checkpoint": str(deployed_path), "deploy_checkpoint_bytes": deployed_path.stat().st_size,
        "real_image": str(image_path), "input_shape": list(image.shape),
        "preprocessing": "OpenCV BGR to RGB, resize 640x640, FP32 / 255; same tensor both paths",
        "numerical_backend": "CPU FP32 with MKLDNN disabled for strict equivalence",
        "train": {"parameters": get_num_params(train_model), "gflops_640": round(get_flops(train_model, imgsz=640), 6)},
        "deploy": {"parameters": get_num_params(deployed), "gflops_640": round(get_flops(deployed, imgsz=640), 6)},
        "dbb_count": 6, "training_branches_removed": True,
        "block_equivalence": {"rtol": 1e-4, "atol": 1e-5, "details": per_block},
        "model_equivalence": {"rtol": 1e-3, "atol": 1e-4, "details": full},
        "deploy_reload_equivalence": {"rtol": 1e-3, "atol": 1e-4, "details": reload_comparison},
        "status": "PASS",
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("STAGE3_DEPLOY_SUMMARY=" + json.dumps({k: v for k, v in result.items() if not k.endswith("equivalence")}, sort_keys=True))


if __name__ == "__main__":
    main()
