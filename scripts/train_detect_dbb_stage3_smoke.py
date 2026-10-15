"""Exactly one smoke-training epoch for canonical stage 3; no prior run reused."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402


def _metrics(value) -> dict[str, float]:
    return {key: float(number) for key, number in value.results_dict.items()}


def main() -> None:
    import ultralytics

    if ultralytics.__version__ != "8.4.153" or Path(ultralytics.__file__).resolve().parent != (ROOT / "third_party" / "ultralytics").resolve():
        raise RuntimeError("Expected project-local Ultralytics 8.4.153")
    if not (ROOT / "outputs" / "stage3_detect_dbb" / "structural_validation.json").is_file():
        raise RuntimeError("Validate stage 3 structurally before the smoke run")
    image_root = ROOT / "data" / "dataset" / "images"
    train_count = sum(1 for p in (image_root / "train").iterdir() if p.is_file())
    val_count = sum(1 for p in (image_root / "val").iterdir() if p.is_file())
    if (train_count, val_count) != (3082, 385):
        raise AssertionError(f"Unexpected dataset image counts: {(train_count, val_count)}")

    project = ROOT / "runs" / "detect"
    run_dir = project / "leyolo_nano_mgconv_dbb_stage3_smoke"
    val_dir = project / "leyolo_nano_mgconv_dbb_stage3_best_val"
    summary_path = ROOT / "outputs" / "stage3_detect_dbb" / "smoke_training.json"
    for target in (run_dir, val_dir, summary_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing stage-3 artifact: {target}")

    model = YOLO(str(ROOT / "models" / "yolov8n_leyolo_nano_mgconv_detect_dbb_stage3.yaml"), task="detect")
    epoch_metrics = model.train(
        data=str(ROOT / "data" / "data.yaml"),
        epochs=1, imgsz=640, batch=4,
        device="cpu", workers=0,
        project=str(project), name=run_dir.name, exist_ok=False,
        pretrained=False, amp=False, val=True, plots=True,
        seed=0, deterministic=True, verbose=True,
    )
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("One-epoch training finished without best.pt and last.pt")
    result = {
        "epochs": 1, "imgsz": 640, "batch": 4,
        "train_images": train_count, "val_images": val_count,
        "training_run": str(run_dir), "best_validation_run": str(val_dir),
        "best_weights": str(best), "last_weights": str(last),
        "train_epoch_metrics": _metrics(epoch_metrics), "training_status": "PASS",
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    best_metrics = YOLO(str(best), task="detect").val(
        data=str(ROOT / "data" / "data.yaml"), split="val",
        imgsz=640, batch=4, device="cpu", workers=0,
        project=str(project), name=val_dir.name, exist_ok=False,
        plots=True, verbose=True,
    )
    result["explicit_best_validation_metrics"] = _metrics(best_metrics)
    result["best_validation_status"] = "PASS"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("STAGE3_SMOKE_SUMMARY=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
