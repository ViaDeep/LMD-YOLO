"""Stage 10: fine-tune both Stage-9 models with street photos from Lima, mixed with part of the Stage-8 train split.

The Lima split is fixed in configs/lima_split.json. Its locked test split, like the China and the other countries'
test splits, is only read by the `test` action, which runs once and refuses to run again.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
import zipfile
from pathlib import Path

import torch
import yaml

import stage6  # also puts the vendored Ultralytics first on sys.path
import stage8
from stage6 import NAMES, ROOT, YOLO, assert_finite_csv, audit_baseline, audit_lmd, metric_dict, sha256

CONFIG_PATH = ROOT / "configs" / "stage10_lima.yaml"
MODELS = ("lmd_yolo", "yolov8n")
DATASET = Path("/content/dataset_stage10")
LIMA_RAW = Path("/content/lima_raw")
LOCKED_SPLIT = "test_lima_bloqueado"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Validation groups. val_rdd is the same set Stage 9 called val_total (China + other countries).
EVAL_GROUPS = {
    "val_total": ("val_lima", "val_china", "val_paises"),
    "val_lima": ("val_lima",),
    "val_rdd": ("val_china", "val_paises"),
    "val_china": ("val_china",),
    "val_paises": ("val_paises",),
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def stage8_root(cfg: dict) -> Path:
    return Path(cfg["stage8_results_root"])


def paths(results_root: Path) -> dict:
    return {
        "manifest": results_root / "dataset_stage10_manifest.json",
        "archive": results_root / "dataset_stage10.zip",
    }


def lima_zip(cfg: dict) -> Path:
    """The single zip in the Lima folder of Drive, whatever its export name."""
    zips = sorted(Path(cfg["lima_dir"]).glob("*.zip"))
    if len(zips) != 1:
        raise FileNotFoundError(f"Se esperaba un solo zip en {cfg['lima_dir']} y hay {len(zips)}: {[z.name for z in zips]}")
    return zips[0]


def check_label(text: str, where: str) -> list[int]:
    classes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{where}:{number}: se esperaban 5 valores YOLO")
        cls, *box = map(float, fields)
        x, y, w, h = box
        if cls != int(cls) or int(cls) not in range(len(NAMES)):
            raise ValueError(f"{where}:{number}: clase fuera de 0-3")
        if not all(0.0 <= v <= 1.0 for v in box) or w <= 0 or h <= 0:
            raise ValueError(f"{where}:{number}: coordenadas fuera de 0-1")
        classes.append(int(cls))
    if not classes:
        raise ValueError(f"{where}: etiqueta vacía")
    return classes


def read_lima(zip_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Images and label files of the Lima export, by file stem, whatever folders the export uses."""
    if not LIMA_RAW.exists():
        with zipfile.ZipFile(zip_path) as bundle:
            bundle.extractall(LIMA_RAW)
    images, labels = {}, {}
    for path in sorted(LIMA_RAW.rglob("*")):
        if not path.is_file() or "__MACOSX" in path.parts:
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images:
                raise ValueError(f"Foto repetida en el zip: {path.stem}")
            images[path.stem] = path
        elif path.suffix == ".txt" and "labels" in (p.lower() for p in path.parts):
            labels[path.stem] = path
    return images, labels


def save_photo(source: Path, target: Path, max_side: int) -> None:
    """Upright (EXIF applied, as the app does) and with the long side at most max_side; boxes are normalized."""
    from PIL import Image, ImageOps

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "JPEG", quality=95)


def prepare(results_root: Path) -> None:
    cfg = load_config()
    p = paths(results_root)
    if p["manifest"].exists():
        raise FileExistsError("El dataset de la etapa 10 ya está preparado; usa la acción restore")
    split = json.loads((ROOT / cfg["lima_split"]).read_text(encoding="utf-8"))
    stage8.restore(stage8_root(cfg), full_hash=True)
    rdd = stage8.paths(stage8_root(cfg))["dataset"]
    source_zip = lima_zip(cfg)
    images, labels = read_lima(source_zip)
    listed = [stem for group in ("train", "val", "test") for stem in split[group]]
    if len(set(listed)) != len(listed):
        raise AssertionError("Una foto aparece en dos grupos de lima_split.json")
    missing = [stem for stem in listed if stem not in images or stem not in labels]
    if missing:
        raise FileNotFoundError(f"Faltan en lima.zip (foto o etiqueta): {missing}")
    left_out = sorted(set(images) - set(listed))
    if DATASET.exists():
        shutil.rmtree(DATASET)
    lima, objects = {}, {}
    for group, target, repeat in (("train", "train", int(cfg["lima_repeat"])), ("val", "val_lima", 1), ("test", LOCKED_SPLIT, 1)):
        counts = [0] * len(NAMES)
        for stem in split[group]:
            text = labels[stem].read_text(encoding="utf-8")
            for cls in check_label(text, str(labels[stem])):
                counts[cls] += 1
            first = DATASET / "images" / target / f"{stem}.jpg"
            save_photo(images[stem], first, int(cfg["lima_max_side"]))
            for copy in range(repeat):
                name = stem if copy == 0 else f"{stem}_r{copy}"
                if copy:
                    shutil.copy2(first, DATASET / "images" / target / f"{name}.jpg")
                (DATASET / "labels" / target).mkdir(parents=True, exist_ok=True)
                (DATASET / "labels" / target / f"{name}.txt").write_text(text, encoding="utf-8")
        lima[target] = len(split[group])
        objects[target] = dict(zip(NAMES, counts))
    rdd_train = sorted((rdd / "images" / "train").glob("*.jpg"))
    sample = random.Random(int(cfg["seed"])).sample(rdd_train, int(cfg["rdd_train_sample"]))
    for image in sorted(sample):
        stage8.place(DATASET, "train", image, (rdd / "labels" / "train" / f"{image.stem}.txt").read_text(encoding="utf-8"), "")
    for split_name in ("val_china", "val_paises"):
        for image in sorted((rdd / "images" / split_name).glob("*.jpg")):
            stage8.place(DATASET, split_name, image, (rdd / "labels" / split_name / f"{image.stem}.txt").read_text(encoding="utf-8"), "")
    sources = {
        "lima": {"photos": lima, "objects": objects, "train_repeat": int(cfg["lima_repeat"]),
                 "left_out_without_split": left_out, "zip_name": source_zip.name, "zip_sha256": sha256(source_zip)},
        "rdd_train_sample": len(sample),
    }
    manifest = stage8.build_manifest(DATASET, sources)
    results_root.mkdir(parents=True, exist_ok=True)
    # The manifest goes last: its presence means the archive in Drive is complete.
    shutil.make_archive(str(p["archive"].with_suffix("")), "zip", DATASET)
    p["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"split_counts": manifest["split_counts"], **sources}, indent=2))


def restore(results_root: Path, full_hash: bool) -> dict:
    p = paths(results_root)
    manifest = json.loads(p["manifest"].read_text(encoding="utf-8"))
    if not DATASET.exists():
        with zipfile.ZipFile(p["archive"]) as bundle:
            bundle.extractall(DATASET)
    return stage8.verify_manifest(DATASET, manifest, full_hash)


def data_yaml(results_root: Path, val_splits: tuple[str, ...], name: str) -> Path:
    target = results_root / f"{name}.yaml"
    payload = {
        "path": str(DATASET),
        "train": "images/train",
        "val": [f"images/{split}" for split in val_splits],
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def train_kwargs(results_root: Path, cfg: dict) -> dict:
    kwargs = stage6.train_kwargs(data_yaml(results_root, EVAL_GROUPS["val_total"], "data_stage10"), cfg)
    # Fine-tuning: keep the Stage-9 weights (stage6 trains from scratch with pretrained=False).
    kwargs["pretrained"] = True
    return kwargs


def evaluate(weights: Path, results_root: Path, out_dir: Path, cfg: dict) -> dict:
    report = {}
    for label, splits in EVAL_GROUPS.items():
        val = YOLO(str(weights), task="detect", verbose=False).val(
            data=str(data_yaml(results_root, splits, f"eval_{label}")), split="val", imgsz=cfg["imgsz"],
            batch=cfg["batch"], device=0, workers=cfg["workers"], project=str(out_dir), name=label,
            exist_ok=False, plots=True, verbose=True,
        )
        report[label] = {
            "metrics": metric_dict(val),
            "per_class": {
                n: {"precision": float(val.box.p[i]), "recall": float(val.box.r[i]),
                    "mAP50": float(val.box.ap50[i]), "mAP50_95": float(val.box.maps[i])}
                for i, n in enumerate(NAMES)
            },
        }
    return report


def init_model(name: str, cfg: dict) -> tuple[YOLO, dict]:
    spec = cfg["init_weights"][name]
    weights = Path(spec["drive_path"])
    if sha256(weights) != spec["sha256"]:
        raise AssertionError(f"El best.pt de la etapa 9 de {name} no coincide con el registrado")
    stage6.seed_everything(int(cfg["seed"]))
    model = YOLO(str(weights), task="detect", verbose=False)
    structure = audit_lmd(model) if name == "lmd_yolo" else audit_baseline(model)
    return model, {"init_weights": str(weights), "init_sha256": spec["sha256"], "structure": structure}


def baseline(results_root: Path) -> None:
    """Stage-9 models on the Stage-10 validation sets: the 'before' column."""
    cfg = load_config()
    restore(results_root, full_hash=False)
    out = results_root / "baseline_stage9"
    if out.exists():
        raise FileExistsError(f"Ya existe {out}")
    report = {}
    for name in MODELS:
        init_model(name, cfg)  # checks the hash
        report[name] = evaluate(Path(cfg["init_weights"][name]["drive_path"]), results_root, out / name, cfg)
    (out / "baseline_stage9.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in MODELS:
        print(f"{name:9s} etapa 9 en val_lima: mAP@50 {report[name]['val_lima']['metrics']['metrics/mAP50(B)']:.3f}")


def smoke(results_root: Path) -> None:
    cfg = load_config()
    dataset = restore(results_root, full_hash=True)
    smoke_root = results_root / "smoke"
    summaries = {}
    for name in MODELS:
        if (smoke_root / name).exists():
            raise FileExistsError(f"Ya existe el smoke de {name}")
        model, _ = init_model(name, cfg)
        kwargs = train_kwargs(results_root, cfg)
        kwargs.update(epochs=cfg["smoke"]["epochs"], fraction=cfg["smoke"]["fraction"], batch=cfg["smoke"]["batch"],
                      project=str(smoke_root), name=name, exist_ok=False)
        started = time.time()
        model.train(**kwargs)
        summaries[name] = {"status": "PASS", "rows": assert_finite_csv(smoke_root / name / "results.csv"), "seconds": time.time() - started}
    gate = {"status": "PASS", "dataset": dataset, "models": summaries, "config_sha256": sha256(CONFIG_PATH),
            "dataset_manifest_sha256": sha256(paths(results_root)["manifest"])}
    (results_root / "smoke_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


def elapsed_seconds(rows: list[dict]) -> float:
    """Training time across sessions: Ultralytics restarts the cumulative `time` column when a run is resumed."""
    total, previous = 0.0, 0.0
    for row in rows:
        current = float(row["time"])
        if current < previous:
            total += previous
        previous = current
    return total + previous


def train_model(name: str, results_root: Path, resume: bool) -> None:
    cfg = load_config()
    gate_path = results_root / "smoke_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    if gate.get("status") != "PASS" or gate["config_sha256"] != sha256(CONFIG_PATH) or gate["dataset_manifest_sha256"] != sha256(paths(results_root)["manifest"]):
        raise RuntimeError("Falta un smoke válido de la etapa 10")
    restore(results_root, full_hash=False)
    project = results_root / "training"
    run_dir = project / name
    last_ckpt = run_dir / "weights" / "last.pt"
    resuming = resume and last_ckpt.is_file() and not (run_dir / "stage10_summary.json").exists()
    if run_dir.exists() and not resuming:
        raise FileExistsError(f"Ya existe la corrida de {name}; usa --resume si se cortó")
    model, provenance = init_model(name, cfg)
    torch.cuda.reset_peak_memory_stats()
    if resuming:
        YOLO(str(last_ckpt), task="detect", verbose=False).train(resume=True)
    else:
        model.train(**train_kwargs(results_root, cfg), epochs=cfg["epochs"], project=str(project), name=name,
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
        "model": name, "status": "COMPLETE", "initialization": "stage9_best", **provenance,
        "config": CONFIG_PATH.name, "config_sha256": sha256(CONFIG_PATH), "resumed": resuming,
        "epochs": len(rows), "best_epoch": int(best_row["epoch"]), "training_seconds": seconds,
        "seconds_per_epoch": seconds / len(rows), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "validation": evaluate(best_pt, results_root, results_root / "best_validation" / name, cfg),
        "best_pt": str(best_pt), "best_pt_sha256": sha256(best_pt), "test_used": False,
    }
    (run_dir / "stage10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def compare(results_root: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    before = json.loads((results_root / "baseline_stage9" / "baseline_stage9.json").read_text(encoding="utf-8"))
    table, frames = {}, {}
    for name in MODELS:
        run = results_root / "training" / name
        after = json.loads((run / "stage10_summary.json").read_text(encoding="utf-8"))
        frames[name] = pd.read_csv(run / "results.csv")
        table[name] = {
            split: {"stage9": before[name][split], "stage10": after["validation"][split]}
            for split in EVAL_GROUPS
        }
        table[name]["summary"] = after
    columns = ["train/box_loss", "train/cls_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, column in zip(axes.flat, columns):
        for name, frame in frames.items():
            ax.plot(frame["epoch"], frame[column], label=name)
        ax.set_title(column); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Etapa 10 (validación: Lima + China + otros países)")
    fig.tight_layout()
    out = results_root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "learning_curves.png", dpi=160)
    payload = {"status": "READY_FOR_TECHNICAL_REVIEW", "test_used": False, **table}
    (out / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name in MODELS:
        for split in ("val_lima", "val_rdd", "val_china", "val_paises", "val_total"):
            b = table[name][split]["stage9"]["metrics"]["metrics/mAP50(B)"]
            a = table[name][split]["stage10"]["metrics"]["metrics/mAP50(B)"]
            print(f"{name:9s} {split:10s} mAP@50 etapa 9 {b:.3f} -> etapa 10 {a:.3f}")


def test(results_root: Path) -> None:
    """The single evaluation of the final models on the locked test splits (Lima, China, other countries)."""
    cfg = load_config()
    out = results_root / "test_final"
    if out.exists():
        raise FileExistsError(f"El TEST ya se evaluó ({out}); no se vuelve a evaluar")
    restore(results_root, full_hash=True)
    stage8.restore(stage8_root(cfg), full_hash=True)
    stage6.verify_dataset(full_hash=True)
    test_splits = {
        "test_lima": DATASET / "images" / LOCKED_SPLIT,
        "test_china": ROOT / "dataset" / "images" / "test",
        "test_paises": stage8.paths(stage8_root(cfg))["dataset"] / "images" / stage8.LOCKED_SPLIT,
    }
    weights = {}
    for name in MODELS:
        summary = json.loads((results_root / "training" / name / "stage10_summary.json").read_text(encoding="utf-8"))
        best_pt = Path(summary["best_pt"])
        if sha256(best_pt) != summary["best_pt_sha256"]:
            raise AssertionError(f"El best.pt de {name} cambió desde el entrenamiento")
        weights[name] = (best_pt, summary["best_pt_sha256"])
    out.mkdir(parents=True)
    report = {"status": "TEST_USED_ONCE", "config_sha256": sha256(CONFIG_PATH)}
    groups = {"test_total": tuple(test_splits), **{label: (label,) for label in test_splits}}
    for name, (best_pt, digest) in weights.items():
        report[name] = {"best_pt_sha256": digest}
        for label, splits in groups.items():
            data = out / f"data_{label}.yaml"
            dirs = [str(test_splits[s]) for s in splits]
            # Ultralytics requires a train key; validation only reads `val`.
            data.write_text(yaml.safe_dump({"path": "/", "train": list(dirs), "val": dirs,
                                            "names": {i: n for i, n in enumerate(NAMES)}}, sort_keys=False), encoding="utf-8")
            val = YOLO(str(best_pt), task="detect", verbose=False).val(
                data=str(data), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"], device=0, workers=cfg["workers"],
                project=str(out / name), name=label, exist_ok=False, plots=True, verbose=True,
            )
            report[name][label] = {
                "images": sum(len(list(test_splits[s].glob("*.jpg"))) for s in splits),
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
    parser.add_argument("action", choices=("prepare", "restore", "baseline", "smoke", "train", "compare", "test"))
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
    if args.action == "prepare":
        prepare(results_root)
    elif args.action == "restore":
        print(json.dumps(restore(results_root, full_hash=True), indent=2))
    elif args.action == "baseline":
        baseline(results_root)
    elif args.action == "smoke":
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
