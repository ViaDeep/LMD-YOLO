"""Validate canonical stage 3 structurally; never train or overwrite stage 2."""

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
from ultralytics.nn.modules import DetectDBB, DiverseBranchBlock, MGConv  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402
from compare_detect_dbb_experimental import captures, compare  # noqa: E402


def main() -> None:
    import ultralytics

    if ultralytics.__version__ != "8.4.153" or Path(ultralytics.__file__).resolve().parent != (ROOT / "third_party" / "ultralytics").resolve():
        raise RuntimeError("Expected project-local Ultralytics 8.4.153")
    stage2 = ROOT / "models" / "yolov8n_leyolo_nano_mgconv_stage2.yaml"
    stage3 = ROOT / "models" / "yolov8n_leyolo_nano_mgconv_detect_dbb_stage3.yaml"
    a6 = ROOT / "models" / "experimental_leyolo_nano_mgconv_detect_dbb_a6.yaml"
    expected = yaml.safe_load(stage2.read_text(encoding="utf-8"))
    expected["head"][-1][2] = "DetectDBB"
    if yaml.safe_load(stage3.read_text(encoding="utf-8")) != expected:
        raise AssertionError("Stage 3 differs from frozen stage 2 beyond the head")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = YOLO(str(stage3), task="detect", verbose=False).model.eval()
    experiment = YOLO(str(a6), task="detect", verbose=False).model.eval()
    if type(model.model[-1]) is not DetectDBB or model.model[-1].nc != 4 or model.model[-1].reg_max != 16:
        raise AssertionError("Unexpected canonical head/classes/reg_max")
    if [(key, tuple(value.shape)) for key, value in model.state_dict().items()] != [
        (key, tuple(value.shape)) for key, value in experiment.state_dict().items()
    ]:
        raise AssertionError("Canonical stage 3 differs structurally from experimental A")
    blocks = [m for m in model.modules() if isinstance(m, DiverseBranchBlock)]
    if len(blocks) != 6 or [int(s) for s in model.stride.tolist()] != [8, 16, 32]:
        raise AssertionError("Expected six DBBs and strides 8/16/32")
    for scale in range(3):
        for towers in (model.model[-1].cv2, model.model[-1].cv3):
            tower = towers[scale]
            if not isinstance(tower[0], DiverseBranchBlock) or isinstance(tower[1], DiverseBranchBlock):
                raise AssertionError("Expected DBB only at first Conv of each tower")
    for layer in (21, 24):
        if not isinstance(model.model[layer], MGConv) or model.model[layer].groups != 2:
            raise AssertionError("Stage-2 neck changed")
    names = list(yaml.safe_load((ROOT / "data" / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    if names != ["D00", "D10", "D20", "D40"]:
        raise AssertionError("Unexpected dataset classes")

    x = torch.randn(1, 3, 640, 640)
    model.train()
    with torch.no_grad():
        train_output = model(x)
    if train_output["boxes"].shape != (1, 64, 8400) or train_output["scores"].shape != (1, 4, 8400):
        raise AssertionError("Unexpected training-mode logits")
    model.eval()
    params_train = get_num_params(model)
    gflops_train = get_flops(model, imgsz=640)
    decoded_before, preds_before, raw_before, features_before = captures(model, x)

    deploy = deepcopy(model).eval()
    for block in deploy.modules():
        if isinstance(block, DiverseBranchBlock):
            block.switch_to_deploy()
    deploy_blocks = [m for m in deploy.modules() if isinstance(m, DiverseBranchBlock)]
    if len(deploy_blocks) != 6 or any(
        not hasattr(m, "dbb_reparam") or any(hasattr(m, k) for k in ("dbb_origin", "dbb_1x1", "dbb_avg", "dbb_1x1_kxk"))
        for m in deploy_blocks
    ):
        raise AssertionError("Deploy DBBs retain training branches")
    decoded_after, preds_after, raw_after, features_after = captures(deploy, x)
    expected_features = {"P3": [1, 64, 80, 80], "P4": [1, 128, 40, 40], "P5": [1, 256, 20, 20]}
    if features_before != expected_features or features_after != expected_features:
        raise AssertionError("Unexpected P3/P4/P5")
    discrepancies = {k: compare(raw_before[k], raw_after[k], 1e-3, 1e-4) for k in raw_before}
    discrepancies["boxes_concat"] = compare(preds_before["boxes"], preds_after["boxes"], 1e-3, 1e-4)
    discrepancies["scores_concat"] = compare(preds_before["scores"], preds_after["scores"], 1e-3, 1e-4)
    discrepancies["decoded"] = compare(decoded_before, decoded_after, 1e-3, 1e-4)

    result = {
        "yaml": str(stage3), "classes": names, "dbb_count": 6,
        "head": type(model.model[-1]).__name__, "matches_experimental_A": True,
        "features": expected_features, "strides": [8, 16, 32],
        "train": {"parameters": params_train, "gflops_640": round(gflops_train, 6)},
        "deploy": {"parameters": get_num_params(deploy), "gflops_640": round(get_flops(deploy, imgsz=640), 6)},
        "training_mode_forward": "PASS", "eval_forward_1x3x640x640": "PASS",
        "training_branches_removed_in_deploy": True,
        "equivalence": {"rtol": 1e-3, "atol": 1e-4, "details": discrepancies},
    }
    if result["train"] != {"parameters": 3_073_884, "gflops_640": 8.56617} or result["deploy"] != {"parameters": 2_265_820, "gflops_640": 6.157107}:
        raise AssertionError("Canonical stage 3 differs from previously validated A measurements")
    output = ROOT / "outputs" / "stage3_detect_dbb"
    output.mkdir(parents=True, exist_ok=True)
    (output / "structural_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "equivalence"}, indent=2))


if __name__ == "__main__":
    main()
