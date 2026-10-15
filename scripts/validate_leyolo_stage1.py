"""Build and structurally compare the two stage-1 LMD_YOLO candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ROOT = ROOT / "third_party"
sys.path.insert(0, str(VENDORED_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


REFERENCE_PARAMS_M = 1.39
REFERENCE_GFLOPS = 3.9
FEATURE_LAYERS = {
    "backbone_p3": 4,
    "backbone_p4": 9,
    "backbone_p5": 14,
    "neck_p3": 20,
    "neck_p4": 23,
    "neck_p5": 26,
}
CANDIDATES = {
    "nano": ROOT / "models" / "yolov8n_leyolo_nano_stage1.yaml",
    "small": ROOT / "models" / "yolov8n_leyolo_small_stage1.yaml",
}


def capture_shapes(model: torch.nn.Module, size: int = 640) -> dict[str, list[int]]:
    """Capture backbone and neck feature-map shapes during a real forward pass."""
    shapes: dict[str, list[int]] = {}
    handles = []

    for name, index in FEATURE_LAYERS.items():
        def hook(_module, _inputs, output, feature_name=name):
            shapes[feature_name] = list(output.shape)

        handles.append(model.model[index].register_forward_hook(hook))

    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, size, size))

    for handle in handles:
        handle.remove()

    if output is None:
        raise RuntimeError("Forward pass returned None")
    return shapes


def validate_candidate(name: str, yaml_path: Path) -> dict:
    """Build one candidate and return reproducible structural measurements."""
    wrapper = YOLO(str(yaml_path), task="detect", verbose=False)
    model = wrapper.model
    shapes = capture_shapes(model)
    params = get_num_params(model)
    gflops = get_flops(model, imgsz=640)
    strides = [int(value) for value in model.stride.tolist()]
    detect = model.model[-1]

    expected_spatial = {
        "backbone_p3": [80, 80],
        "backbone_p4": [40, 40],
        "backbone_p5": [20, 20],
        "neck_p3": [80, 80],
        "neck_p4": [40, 40],
        "neck_p5": [20, 20],
    }
    for feature, spatial in expected_spatial.items():
        if shapes[feature][-2:] != spatial:
            raise AssertionError(f"{name}: {feature} has {shapes[feature]}, expected spatial {spatial}")
    if strides != [8, 16, 32]:
        raise AssertionError(f"{name}: unexpected detection strides {strides}")
    if detect.nc != 4:
        raise AssertionError(f"{name}: Detect.nc={detect.nc}, expected 4")
    if gflops <= 0:
        raise AssertionError(f"{name}: FLOPs profiler returned {gflops}")

    params_m = params / 1e6
    distance = abs(params_m - REFERENCE_PARAMS_M) / REFERENCE_PARAMS_M
    distance += abs(gflops - REFERENCE_GFLOPS) / REFERENCE_GFLOPS
    return {
        "candidate": name,
        "yaml": str(yaml_path.relative_to(ROOT)),
        "parameters": params,
        "parameters_m": round(params_m, 6),
        "gflops_640": round(gflops, 6),
        "detection_strides": strides,
        "detect_classes": detect.nc,
        "feature_shapes": shapes,
        "reference_distance": round(distance, 6),
    }


def write_report(results: list[dict]) -> None:
    """Write machine-readable and human-readable comparison reports."""
    output_dir = ROOT / "outputs" / "stage1_leyolo"
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda item: item["reference_distance"])
    payload = {
        "reference": {"parameters_m": REFERENCE_PARAMS_M, "gflops_640": REFERENCE_GFLOPS},
        "selection_rule": "lowest normalized absolute distance; reference only, no architecture tuning",
        "most_consistent": ordered[0]["candidate"],
        "results": results,
    }
    (output_dir / "structural_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Stage 1 - LeYOLO structural comparison",
        "",
        "The paper values (1.39 M parameters and 3.9 GFLOPs at 640) are references only.",
        "No layer or channel was tuned to match them.",
        "",
        "| Candidate | Parameters (M) | GFLOPs | Strides | Backbone P3/P4/P5 |",
        "|---|---:|---:|---|---|",
    ]
    for result in results:
        feature_shapes = result["feature_shapes"]
        backbone = " / ".join(
            "x".join(map(str, feature_shapes[f"backbone_{level}"][1:])) for level in ("p3", "p4", "p5")
        )
        lines.append(
            f"| {result['candidate']} | {result['parameters_m']:.6f} | "
            f"{result['gflops_640']:.6f} | {result['detection_strides']} | {backbone} |"
        )
    lines.extend(
        [
            "",
            f"Most consistent by the declared rule: **{ordered[0]['candidate']}**.",
            "",
            "This selection does not prove which unpublished configuration the LMD_YOLO authors used.",
        ]
    )
    (output_dir / "structural_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import ultralytics

    expected_package = ROOT / "third_party" / "ultralytics"
    actual_package = Path(ultralytics.__file__).resolve().parent
    if actual_package != expected_package.resolve():
        raise RuntimeError(f"Expected vendored Ultralytics at {expected_package}, got {actual_package}")
    if ultralytics.__version__ != "8.4.153":
        raise RuntimeError(f"Expected Ultralytics 8.4.153, got {ultralytics.__version__}")

    results = [validate_candidate(name, path) for name, path in CANDIDATES.items()]
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

