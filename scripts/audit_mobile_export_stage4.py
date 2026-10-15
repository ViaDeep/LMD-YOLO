"""Read-only structural audit and optional FP32 ONNX experiment for stage 4.

The ONNX experiment deliberately uses a copied checkpoint in outputs/, static
1x3x640x640 input, no NMS, no graph simplification, and no quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import (  # noqa: E402
    DetectDBB,
    DiverseBranchBlock,
    LeYOLOConv,
    LeYOLOInvertedBottleneck,
    MGConv,
)
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


CHECKPOINT = ROOT / "runs/detect/leyolo_nano_mgconv_dbb_stage3_deploy/weights/best_deploy_fp32.pt"
IMAGE = ROOT / "data/dataset/images/val/China_Drone_000001.jpg"
OUTPUT_DIR = ROOT / "outputs/stage4_mobile_export_audit"
STRUCTURE_REPORT = OUTPUT_DIR / "deploy_structure.json"
ONNX_DIR = OUTPUT_DIR / "onnx_fp32_640"
ONNX_REPORT = OUTPUT_DIR / "onnx_fp32_640_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_tensor() -> torch.Tensor:
    bgr = cv2.imread(str(IMAGE), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(IMAGE)
    rgb = cv2.cvtColor(cv2.resize(bgr, (640, 640)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)


def custom_module_counts(model: torch.nn.Module) -> dict[str, int]:
    counts = Counter(
        f"{type(module).__module__}.{type(module).__name__}"
        for module in model.modules()
        if type(module).__module__.startswith("ultralytics.nn.modules")
    )
    return dict(sorted(counts.items()))


def inspect_deploy() -> dict:
    import ultralytics

    if ultralytics.__version__ != "8.4.153":
        raise RuntimeError(f"Expected Ultralytics 8.4.153, found {ultralytics.__version__}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    loaded = YOLO(str(CHECKPOINT), task="detect")
    model = loaded.model.float().eval()
    dbbs = [module for module in model.modules() if isinstance(module, DiverseBranchBlock)]
    mgconvs = [module for module in model.modules() if isinstance(module, MGConv)]
    leyolo_convs = [module for module in model.modules() if isinstance(module, LeYOLOConv)]
    leyolo_bottlenecks = [module for module in model.modules() if isinstance(module, LeYOLOInvertedBottleneck)]
    training_attrs = ("dbb_origin", "dbb_1x1", "dbb_avg", "dbb_1x1_kxk")

    if type(model.model[-1]) is not DetectDBB:
        raise AssertionError(f"Unexpected head: {type(model.model[-1]).__name__}")
    if len(dbbs) != 6:
        raise AssertionError(f"Expected 6 DBBs, found {len(dbbs)}")
    if any(not hasattr(module, "dbb_reparam") for module in dbbs):
        raise AssertionError("At least one DBB lacks its deploy convolution")
    retained = {
        str(index): [name for name in training_attrs if hasattr(module, name)]
        for index, module in enumerate(dbbs)
        if any(hasattr(module, name) for name in training_attrs)
    }
    if retained:
        raise AssertionError(f"DBB training branches retained: {retained}")

    x = image_tensor()
    with torch.inference_mode():
        output = model(x)[0]
    result = {
        "status": "PASS",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_path": str(Path(ultralytics.__file__).resolve()),
        "torch_version": torch.__version__,
        "dtype": str(next(model.parameters()).dtype),
        "parameters": get_num_params(model),
        "gflops_640": round(get_flops(model, imgsz=640), 6),
        "head_type": f"{type(model.model[-1]).__module__}.{type(model.model[-1]).__name__}",
        "classes": model.names,
        "strides": [int(v) for v in model.stride.tolist()],
        "pre_nms_output_shape": list(output.shape),
        "dbb": {
            "count": len(dbbs),
            "all_reparameterized": True,
            "training_branches_present": False,
            "effective_children": [
                {
                    "reparam_type": type(module.dbb_reparam).__name__,
                    "kernel": list(module.dbb_reparam.kernel_size),
                    "stride": list(module.dbb_reparam.stride),
                    "padding": list(module.dbb_reparam.padding),
                    "groups": module.dbb_reparam.groups,
                    "nonlinearity": type(module.nonlinear).__name__,
                }
                for module in dbbs
            ],
        },
        "custom_modules": {
            "MGConv": len(mgconvs),
            "LeYOLOConv": len(leyolo_convs),
            "LeYOLOInvertedBottleneck": len(leyolo_bottlenecks),
            "DetectDBB": 1,
            "DiverseBranchBlock": len(dbbs),
        },
        "mgconv_effective_ops": [
            {
                "main": ["Conv2d", "BatchNorm2d", type(module.main.act).__name__],
                "main_groups": module.main.conv.groups,
                "shortcut": "Conv2d",
                "merge": "Add",
            }
            for module in mgconvs
        ],
        "leyolo_effective_ops": {
            "conv_blocks": ["Conv2d", "BatchNorm2d", "SiLU/ReLU6/GELU/ELU/Hardswish"],
            "bottlenecks": ["pointwise Conv", "depthwise Conv", "pointwise Conv", "BatchNorm", "optional Add"],
            "observed_activations": dict(sorted(Counter(type(module.act).__name__ for module in leyolo_convs).items())),
            "observed_conv_groups": sorted({module.conv.groups for module in leyolo_convs}),
        },
        "all_project_ultralytics_module_types": custom_module_counts(model),
    }
    return result


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise AssertionError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    absolute = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    relative = absolute / np.maximum(np.abs(reference.astype(np.float64)), 1e-8)
    return {
        "shape": list(reference.shape),
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "max_rel_eps_1e-8": float(relative.max()),
        "mean_rel_eps_1e-8": float(relative.mean()),
        "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(reference, candidate, rtol=1e-4, atol=1e-5)),
        "allclose_rtol_1e-3_atol_1e-4": bool(np.allclose(reference, candidate, rtol=1e-3, atol=1e-4)),
    }


def export_and_validate_onnx(structure: dict) -> dict:
    import onnx
    import onnxruntime as ort

    if ONNX_DIR.exists() or ONNX_REPORT.exists():
        raise FileExistsError("Refusing to overwrite an existing ONNX experiment")
    source_hash = sha256(CHECKPOINT)
    ONNX_DIR.mkdir(parents=True, exist_ok=False)
    copied_checkpoint = ONNX_DIR / "best_deploy_fp32_source_copy.pt"
    shutil.copy2(CHECKPOINT, copied_checkpoint)

    # Disable dependency auto-installation so this run cannot silently add tools.
    os.environ["YOLO_AUTOINSTALL"] = "False"
    exported = YOLO(str(copied_checkpoint), task="detect").export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=False,
        half=False,
        quantize=None,
        nms=False,
        device="cpu",
    )
    onnx_path = Path(exported).resolve()
    if onnx_path.parent != ONNX_DIR.resolve():
        raise AssertionError(f"ONNX escaped experiment directory: {onnx_path}")

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    operators = Counter(node.op_type for node in graph.graph.node)
    domains = Counter(node.domain or "ai.onnx" for node in graph.graph.node)
    custom_domains = {name: count for name, count in domains.items() if name != "ai.onnx"}

    x = image_tensor()
    torch_model = YOLO(str(CHECKPOINT), task="detect").model.float().eval()
    with torch.inference_mode():
        torch_output = torch_model(x)[0].cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: x.cpu().numpy()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ort_outputs = session.run(None, ort_inputs)
    if len(ort_outputs) != 1:
        raise AssertionError(f"Expected one ONNX output, got {len(ort_outputs)}")

    result = {
        "status": "PASS",
        "configuration": {
            "precision": "FP32",
            "imgsz": 640,
            "batch": 1,
            "dynamic": False,
            "simplify": False,
            "nms": False,
            "quantization": None,
        },
        "source_checkpoint": str(CHECKPOINT),
        "source_sha256_before": source_hash,
        "source_sha256_after": sha256(CHECKPOINT),
        "source_unchanged": sha256(CHECKPOINT) == source_hash,
        "copied_checkpoint": str(copied_checkpoint),
        "onnx_file": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": sha256(onnx_path),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "ir_version": graph.ir_version,
        "opset_imports": {item.domain or "ai.onnx": item.version for item in graph.opset_import},
        "checker": "PASS",
        "graph_nodes": len(graph.graph.node),
        "operators": dict(sorted(operators.items())),
        "domains": dict(sorted(domains.items())),
        "custom_operator_domains": custom_domains,
        "inputs": [
            {"name": value.name, "shape": [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]}
            for value in graph.graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]}
            for value in graph.graph.output
        ],
        "runtime": {
            "providers_requested": ["CPUExecutionProvider"],
            "providers_active": session.get_providers(),
            "warnings": [str(item.message) for item in caught],
        },
        "test_image": str(IMAGE),
        "preprocessing": "OpenCV BGR->RGB, direct resize 640x640, NCHW FP32 / 255",
        "comparison_pre_nms_decoded": compare_arrays(torch_output, ort_outputs[0]),
        "structure_report_status": structure["status"],
    }
    if not result["source_unchanged"]:
        raise AssertionError("Source checkpoint changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", action="store_true", help="Also perform the single permitted ONNX experiment")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    structure = inspect_deploy()
    if STRUCTURE_REPORT.exists():
        recorded = json.loads(STRUCTURE_REPORT.read_text(encoding="utf-8"))
        normalized = json.loads(json.dumps(structure))
        if recorded != normalized:
            raise AssertionError("Existing structure report differs from the current read-only audit")
    else:
        STRUCTURE_REPORT.write_text(json.dumps(structure, indent=2), encoding="utf-8")
    print("STRUCTURE=" + json.dumps(structure, sort_keys=True))

    if args.onnx:
        result = export_and_validate_onnx(structure)
        ONNX_REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("ONNX=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
