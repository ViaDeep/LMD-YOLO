"""Run the single-epoch LeYOLO-Nano stage-1 smoke training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    model = YOLO(str(ROOT / "models" / "yolov8n_leyolo_nano_stage1.yaml"), task="detect")
    metrics = model.train(
        data=str(ROOT / "data" / "data.yaml"),
        epochs=1,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs" / "detect"),
        name="leyolo_nano_stage1_smoke",
        exist_ok=False,
        pretrained=False,
        amp=False,
        val=True,
        plots=True,
        seed=0,
        deterministic=True,
        verbose=True,
    )
    summary = {
        "save_dir": str(metrics.save_dir),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
    }
    print("SMOKE_TRAINING_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
