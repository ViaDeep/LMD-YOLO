"""One-epoch functional smoke training of canonical stage-2 MGConv.

The run and explicit best.pt validation use distinct output directories.
This is not an accuracy evaluation or a Detect_DBB implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402


def summary(metrics) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.results_dict.items()}


def main() -> None:
    project = ROOT / "runs" / "detect"
    run_dir = project / "leyolo_nano_mgconv_stage2_smoke"
    val_dir = project / "leyolo_nano_mgconv_stage2_best_val"
    if run_dir.exists() or val_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing stage-2 results: {run_dir} or {val_dir}")

    model = YOLO(str(ROOT / "models" / "yolov8n_leyolo_nano_mgconv_stage2.yaml"), task="detect")
    train_metrics = model.train(
        data=str(ROOT / "data" / "data.yaml"),
        epochs=1,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(project),
        name=run_dir.name,
        exist_ok=False,
        pretrained=False,
        amp=False,
        val=True,
        plots=True,
        seed=0,
        deterministic=True,
        verbose=True,
    )
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("Training completed without both best.pt and last.pt")

    # Training already validates at epoch end; this second call verifies the
    # serialized best.pt can be loaded and validated independently.
    best_metrics = YOLO(str(best_path), task="detect").val(
        data=str(ROOT / "data" / "data.yaml"),
        split="val",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(project),
        name=val_dir.name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )
    result = {
        "training_run": str(run_dir),
        "best_validation_run": str(val_dir),
        "best_weights": str(best_path),
        "last_weights": str(last_path),
        "train_epoch_metrics": summary(train_metrics),
        "explicit_best_validation_metrics": summary(best_metrics),
        "training_status": "PASS",
        "best_validation_status": "PASS",
    }
    output = ROOT / "outputs" / "stage2_mgconv" / "smoke_training.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("STAGE2_SMOKE_SUMMARY=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
