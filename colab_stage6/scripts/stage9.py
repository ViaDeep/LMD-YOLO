"""Stage 9: final training. Both models continue from their Stage-8 best.pt on the Stage-8 dataset.

The locked test splits (China test and the other countries' test) are only read by the `test` action, which runs
once and refuses to run again.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml

import stage6  # also puts the vendored Ultralytics first on sys.path
import stage8
from stage6 import NAMES, ROOT, YOLO, assert_finite_csv, audit_baseline, audit_lmd, metric_dict, sha256

CONFIG_PATH = ROOT / "configs" / "stage9_final.yaml"
MODELS = ("lmd_yolo", "yolov8n")
TEST_SPLITS = {
    "test_china": ROOT / "dataset" / "images" / "test",
    "test_paises": stage8.paths(Path("/"))["dataset"] / "images" / stage8.LOCKED_SPLIT,
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def dataset_root(cfg: dict) -> Path:
    return Path(cfg["dataset_results_root"])


def restore(cfg: dict, full_hash: bool) -> dict:
    return stage8.restore(dataset_root(cfg), full_hash)


def elapsed_seconds(rows: list[dict]) -> float:
    """Training time across sessions: Ultralytics restarts the cumulative `time` column when a run is resumed."""
    total, previous = 0.0, 0.0
    for row in rows:
        current = float(row["time"])
        if current < previous:
            total += previous
        previous = current
    return total + previous


def init_model(name: str, cfg: dict) -> tuple[YOLO, dict]:
    spec = cfg["init_weights"][name]
    weights = Path(spec["drive_path"])
    if sha256(weights) != spec["sha256"]:
        raise AssertionError(f"El best.pt de la etapa 8 de {name} no coincide con el registrado")
    stage6.seed_everything(int(cfg["seed"]))
    model = YOLO(str(weights), task="detect", verbose=False)
    structure = audit_lmd(model) if name == "lmd_yolo" else audit_baseline(model)
    return model, {"init_weights": str(weights), "init_sha256": spec["sha256"], "structure": structure}


def smoke(results_root: Path) -> None:
    cfg = load_config()
    dataset = restore(cfg, full_hash=True)
    smoke_root = results_root / "smoke"
    summaries = {}
    for name in MODELS:
        if (smoke_root / name).exists():
            raise FileExistsError(f"Ya existe el smoke de {name}")
        model, _ = init_model(name, cfg)
        kwargs = stage8.train_kwargs(results_root, cfg)
        kwargs.update(epochs=cfg["smoke"]["epochs"], fraction=cfg["smoke"]["fraction"], batch=cfg["smoke"]["batch"],
                      project=str(smoke_root), name=name, exist_ok=False)
        started = time.time()
        model.train(**kwargs)
        summaries[name] = {"status": "PASS", "rows": assert_finite_csv(smoke_root / name / "results.csv"), "seconds": time.time() - started}
    gate = {"status": "PASS", "dataset": dataset, "models": summaries, "config_sha256": sha256(CONFIG_PATH),
            "dataset_manifest_sha256": sha256(stage8.paths(dataset_root(cfg))["manifest"])}
    (results_root / "smoke_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


def train_model(name: str, results_root: Path, resume: bool) -> None:
    cfg = load_config()
    gate_path = results_root / "smoke_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    manifest = stage8.paths(dataset_root(cfg))["manifest"]
    if gate.get("status") != "PASS" or gate["config_sha256"] != sha256(CONFIG_PATH) or gate["dataset_manifest_sha256"] != sha256(manifest):
        raise RuntimeError("Falta un smoke válido de la etapa 9")
    restore(cfg, full_hash=False)
    project = results_root / "training"
    run_dir = project / name
    last_ckpt = run_dir / "weights" / "last.pt"
    resuming = resume and last_ckpt.is_file() and not (run_dir / "stage9_summary.json").exists()
    if run_dir.exists() and not resuming:
        raise FileExistsError(f"Ya existe la corrida de {name}; usa --resume si se cortó")
    model, provenance = init_model(name, cfg)
    torch.cuda.reset_peak_memory_stats()
    if resuming:
        YOLO(str(last_ckpt), task="detect", verbose=False).train(resume=True)
    else:
        model.train(**stage8.train_kwargs(results_root, cfg), epochs=cfg["epochs"], project=str(project), name=name,
                    exist_ok=False, save_period=5)
    csv_path = run_dir / "results.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if assert_finite_csv(csv_path) != cfg["epochs"]:
        raise AssertionError(f"Se esperaban {cfg['epochs']} épocas en {csv_path}")
    best_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    best_pt = run_dir / "weights" / "best.pt"
    if name == "lmd_yolo":
        audit_lmd(YOLO(str(best_pt), task="detect", verbose=False))
    seconds = elapsed_seconds(rows)
    summary = {
        "model": name, "status": "COMPLETE", "initialization": "stage8_best", **provenance,
        "config": CONFIG_PATH.name, "config_sha256": sha256(CONFIG_PATH), "resumed": resuming,
        "epochs": len(rows), "best_epoch": int(best_row["epoch"]), "training_seconds": seconds,
        "seconds_per_epoch": seconds / len(rows), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "validation": stage8.evaluate(best_pt, results_root, results_root / "best_validation" / name, cfg),
        "best_pt": str(best_pt), "best_pt_sha256": sha256(best_pt), "test_used": False,
    }
    (run_dir / "stage9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def compare(results_root: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    cfg = load_config()
    stage8_training = dataset_root(cfg) / "training"
    table, frames = {}, {}
    for name in MODELS:
        before = json.loads((stage8_training / name / "stage8_summary.json").read_text(encoding="utf-8"))
        run = results_root / "training" / name
        after = json.loads((run / "stage9_summary.json").read_text(encoding="utf-8"))
        # Stage-8 epochs 1-50 followed by Stage-9 epochs 51-150, to show the whole fine-tune in one curve.
        first = pd.read_csv(stage8_training / name / "results.csv")
        second = pd.read_csv(run / "results.csv")
        second["epoch"] += len(first)
        frames[name] = (first, second)
        table[name] = {
            split: {"stage8": before["validation"][split]["metrics"], "stage9": after["validation"][split]["metrics"]}
            for split in ("val_total", "val_china", "val_paises")
        }
        table[name]["summary"] = after
    columns = ["train/box_loss", "train/cls_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, column in zip(axes.flat, columns):
        for name, (first, second) in frames.items():
            line, = ax.plot(first["epoch"], first[column], label=f"{name} etapa 8")
            ax.plot(second["epoch"], second[column], color=line.get_color(), linestyle="--", label=f"{name} etapa 9")
        ax.axvline(len(frames[MODELS[0]][0]) + 0.5, color="gray", linewidth=0.8)
        ax.set_title(column); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = results_root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "learning_curves.png", dpi=160)
    payload = {"status": "READY_FOR_TECHNICAL_REVIEW", "test_used": False, **table}
    (out / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name in MODELS:
        for split in ("val_china", "val_paises", "val_total"):
            b, a = table[name][split]["stage8"]["metrics/mAP50(B)"], table[name][split]["stage9"]["metrics/mAP50(B)"]
            print(f"{name:9s} {split:10s} mAP@50 etapa 8 {b:.3f} -> etapa 9 {a:.3f}")


def test(results_root: Path) -> None:
    """The single evaluation of the final models on the locked test splits."""
    cfg = load_config()
    out = results_root / "test_final"
    if out.exists():
        raise FileExistsError(f"El TEST ya se evaluó ({out}); no se vuelve a evaluar")
    restore(cfg, full_hash=True)
    stage6.verify_dataset(full_hash=True)
    weights = {}
    for name in MODELS:
        summary = json.loads((results_root / "training" / name / "stage9_summary.json").read_text(encoding="utf-8"))
        best_pt = Path(summary["best_pt"])
        if sha256(best_pt) != summary["best_pt_sha256"]:
            raise AssertionError(f"El best.pt de {name} cambió desde el entrenamiento")
        weights[name] = (best_pt, summary["best_pt_sha256"])
    out.mkdir(parents=True)
    report = {"status": "TEST_USED_ONCE", "config_sha256": sha256(CONFIG_PATH)}
    groups = {"test_total": tuple(TEST_SPLITS), "test_china": ("test_china",), "test_paises": ("test_paises",)}
    for name, (best_pt, digest) in weights.items():
        report[name] = {"best_pt_sha256": digest}
        for label, splits in groups.items():
            data = out / f"data_{label}.yaml"
            dirs = [str(TEST_SPLITS[s]) for s in splits]
            # Ultralytics requires a train key; validation only reads `val`.
            data.write_text(yaml.safe_dump({"path": "/", "train": list(dirs), "val": dirs,
                                            "names": {i: n for i, n in enumerate(NAMES)}}, sort_keys=False), encoding="utf-8")
            val = YOLO(str(best_pt), task="detect", verbose=False).val(
                data=str(data), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"], device=0, workers=cfg["workers"],
                project=str(out / name), name=label, exist_ok=False, plots=True, verbose=True,
            )
            report[name][label] = {
                "images": sum(len(list(TEST_SPLITS[s].glob("*.jpg"))) for s in splits),
                "metrics": metric_dict(val),
                "per_class": {
                    n: {"precision": float(val.box.p[i]), "recall": float(val.box.r[i]),
                        "mAP50": float(val.box.ap50[i]), "mAP50_95": float(val.box.maps[i])}
                    for i, n in enumerate(NAMES)
                },
            }
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in MODELS:
        for label in groups:
            m = report[name][label]["metrics"]
            print(f"{name:9s} {label:11s} mAP@50 {m['metrics/mAP50(B)']:.3f} mAP@50:95 {m['metrics/mAP50-95(B)']:.3f}")


def main() -> None:
    global CONFIG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "train", "compare", "test"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirmo-modelo-final", action="store_true",
                        help="test: confirma que estos son los modelos finales; el TEST se evalúa una sola vez")
    args = parser.parse_args()
    CONFIG_PATH = args.config.expanduser().resolve()
    stage6.CONFIG_PATH = stage8.CONFIG_PATH = CONFIG_PATH
    results_root = args.results_root.expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    if args.action == "smoke":
        smoke(results_root)
    elif args.action == "train":
        if not args.model:
            parser.error("train requiere --model")
        train_model(args.model, results_root, args.resume)
    elif args.action == "compare":
        compare(results_root)
    else:
        if not args.confirmo_modelo_final:
            parser.error("test requiere --confirmo-modelo-final")
        test(results_root)


if __name__ == "__main__":
    main()
