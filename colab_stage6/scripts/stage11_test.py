"""Single final evaluation on the locked TEST splits (Lima, China, other countries).

Final models: the 50/50 weight averages of Stage 10 and Stage 11 (soup/weights/*_alpha50.pt), the LMD-YOLO one is the
model in the app. The Stage-9 models (before the Lima fine-tuning) are evaluated in the same run as the 'before'
reference. The TEST is read only here; the script refuses to run a second time. Results go to
<results-root>/test_modelo_final/test_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import stage11
from stage6 import NAMES, ROOT, YOLO, metric_dict, sha256

FINAL = {"lmd_yolo": "lmd_yolo_alpha50", "yolov8n": "yolov8n_alpha50"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=stage11.CONFIG_PATH)
    parser.add_argument("--confirmo-modelo-final", action="store_true",
                        help="confirma que estos son los modelos finales; el TEST se evalúa una sola vez")
    args = parser.parse_args()
    if not args.confirmo_modelo_final:
        parser.error("requiere --confirmo-modelo-final")
    stage11.CONFIG_PATH = stage11.stage6.CONFIG_PATH = stage11.stage8.CONFIG_PATH = args.config.expanduser().resolve()
    root = args.results_root.expanduser().resolve()
    cfg = stage11.load_config()
    out = root / "test_modelo_final"
    if out.exists():
        raise FileExistsError(f"El TEST ya se evaluó ({out}); no se vuelve a evaluar")

    soup = json.loads((root / "soup" / "soup_summary.json").read_text(encoding="utf-8"))
    weights = {}
    for name, key in FINAL.items():
        path = Path(soup[key]["weights"])
        if sha256(path) != soup[key]["sha256"]:
            raise AssertionError(f"El modelo final {key} cambió desde que se evaluó en validación")
        weights[f"{name}_final"] = (path, soup[key]["sha256"])
        stage11.init_model(name, cfg)  # checks the Stage-9 hash
        before = Path(cfg["init_weights"][name]["drive_path"])
        weights[f"{name}_etapa9"] = (before, sha256(before))

    stage11.restore(root, full_hash=True)
    stage11.stage8.restore(stage11.stage8_root(cfg), full_hash=True)
    stage11.stage6.verify_dataset(full_hash=True)
    splits = {
        "test_lima": stage11.DATASET / "images" / stage11.LOCKED_SPLIT,
        "test_china": ROOT / "dataset" / "images" / "test",
        "test_paises": stage11.stage8.paths(stage11.stage8_root(cfg))["dataset"] / "images" / stage11.stage8.LOCKED_SPLIT,
    }
    for label, folder in splits.items():
        count = len(list(folder.glob("*.jpg")))
        if count == 0:
            raise AssertionError(f"No hay imágenes en {folder}")
        print(f"{label}: {count} imágenes")
    groups = {"test_rdd": ("test_china", "test_paises"), **{label: (label,) for label in splits}}

    out.mkdir(parents=True)
    report = {"status": "TEST_USED_ONCE", "config_sha256": sha256(stage11.CONFIG_PATH)}
    for model, (path, digest) in weights.items():
        report[model] = {"weights": str(path), "sha256": digest}
        for label, members in groups.items():
            dirs = [str(splits[s]) for s in members]
            data = out / f"data_{label}.yaml"
            # Ultralytics requires a train key; validation only reads `val`.
            data.write_text(yaml.safe_dump({"path": "/", "train": dirs, "val": dirs,
                                            "names": {i: n for i, n in enumerate(NAMES)}}, sort_keys=False),
                            encoding="utf-8")
            val = YOLO(str(path), task="detect", verbose=False).val(
                data=str(data), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"], device=0, workers=cfg["workers"],
                project=str(out / model), name=label, exist_ok=False, plots=True, verbose=False,
            )
            report[model][label] = {
                "images": sum(len(list(splits[s].glob("*.jpg"))) for s in members),
                "metrics": metric_dict(val),
                "per_class": {
                    n: {"precision": float(val.box.p[i]), "recall": float(val.box.r[i]),
                        "mAP50": float(val.box.ap50[i]), "mAP50_95": float(val.box.maps[i])}
                    for i, n in enumerate(NAMES)
                },
            }
            (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for model in weights:
        for label in groups:
            m = report[model][label]["metrics"]
            print(f"{model:16s} {label:11s} mAP@50 {m['metrics/mAP50(B)']:.3f} mAP@50:95 {m['metrics/mAP50-95(B)']:.3f} "
                  f"P {m['metrics/precision(B)']:.3f} R {m['metrics/recall(B)']:.3f}")


if __name__ == "__main__":
    main()
