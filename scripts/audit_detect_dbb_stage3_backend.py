"""Read-only MKLDNN vs native-CPU numerical audit of trained DBB deployment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402
from convert_validate_detect_dbb_stage3 import capture  # noqa: E402
from compare_detect_dbb_experimental import errors  # noqa: E402


def discrepancy(left: torch.Tensor, right: torch.Tensor, rtol: float, atol: float) -> dict:
    result = errors(left, right)
    result["failed_elements"] = int((~torch.isclose(left, right, rtol=rtol, atol=atol)).sum())
    result["passed"] = result["failed_elements"] == 0
    return result


def main() -> None:
    torch.set_num_threads(4)
    output = ROOT / "outputs" / "stage3_detect_dbb" / "backend_diagnostic.json"
    if output.exists():
        raise FileExistsError(output)
    best = ROOT / "runs" / "detect" / "leyolo_nano_mgconv_dbb_stage3_smoke" / "weights" / "best.pt"
    deploy = ROOT / "runs" / "detect" / "leyolo_nano_mgconv_dbb_stage3_deploy" / "weights" / "best_deploy_fp32.pt"
    image_path = ROOT / "data" / "dataset" / "images" / "val" / "China_Drone_000001.jpg"
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(image_path)), (640, 640)), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)
    original = YOLO(str(best), task="detect").model.float().eval()
    converted = YOLO(str(deploy), task="detect").model.float().eval()
    result = {"image": str(image_path), "backends": {}}
    prior = torch.backends.mkldnn.enabled
    try:
        for enabled in (False, True):
            torch.backends.mkldnn.enabled = enabled
            left, left_pred, left_raw, left_features, inputs = capture(original, image)
            right, right_pred, right_raw, right_features, _ = capture(converted, image)
            block_results = {}
            for scale in range(3):
                for branch in ("cv2", "cv3"):
                    key = f"P{scale + 3}_{branch}"
                    with torch.no_grad():
                        a = getattr(original.model[-1], branch)[scale][0](inputs[key])
                        b = getattr(converted.model[-1], branch)[scale][0](inputs[key])
                    block_results[key] = discrepancy(a, b, 1e-4, 1e-5)
            model_results = {key: discrepancy(left_raw[key], right_raw[key], 1e-3, 1e-4) for key in left_raw}
            model_results["boxes_concat"] = discrepancy(left_pred["boxes"], right_pred["boxes"], 1e-3, 1e-4)
            model_results["scores_concat"] = discrepancy(left_pred["scores"], right_pred["scores"], 1e-3, 1e-4)
            model_results["decoded"] = discrepancy(left, right, 1e-3, 1e-4)
            model_results.update({f"{k}_feature": discrepancy(left_features[k], right_features[k], 1e-3, 1e-4) for k in left_features})
            result["backends"]["mkldnn_on" if enabled else "mkldnn_off"] = {
                "blocks": block_results, "model": model_results,
                "all_blocks_pass": all(v["passed"] for v in block_results.values()),
                "full_model_pass": all(v["passed"] for v in model_results.values()),
            }
    finally:
        torch.backends.mkldnn.enabled = prior
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: {"all_blocks_pass": v["all_blocks_pass"], "full_model_pass": v["full_model_pass"], "failed_blocks": {k: z["failed_elements"] for k, z in v["blocks"].items() if not z["passed"]}} for name, v in result["backends"].items()}, indent=2))


if __name__ == "__main__":
    main()
