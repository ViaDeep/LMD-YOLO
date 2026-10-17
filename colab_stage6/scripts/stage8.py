"""Stage 8: fine-tune both Stage-7 models on RDD2022 China plus street-level images from other countries."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import torch
import yaml

import stage6  # also puts the vendored Ultralytics first on sys.path
from stage6 import NAMES, ROOT, YOLO, assert_finite_csv, audit_baseline, audit_lmd, metric_dict, sha256

CONFIG_PATH = ROOT / "configs" / "stage8_paises.yaml"
CHINA_DATASET = ROOT / "dataset"
CLASS_MAP = {name: i for i, name in enumerate(NAMES)}
S3_URL = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_{country}.zip"
FIGSHARE_FILES = "https://api.figshare.com/v2/articles/21431547/files"
# Splits that go into the data YAML. The China test split and the countries' test split are never listed.
TRAIN_VAL_SPLITS = ("train", "val_china", "val_paises")
LOCKED_SPLIT = "test_paises_bloqueado"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Descargando {url}")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    if not zipfile.is_zipfile(partial):
        partial.unlink()
        raise RuntimeError(f"La descarga no es un ZIP válido: {url}")
    partial.rename(target)


def fetch_country_zip(country: str, cache: Path, work: Path) -> Path:
    """Return a local ZIP for one country: Drive cache, then the official S3 link, then FigShare."""
    cached = cache / f"RDD2022_{country}.zip"
    if cached.is_file() and zipfile.is_zipfile(cached):
        return cached
    try:
        download(S3_URL.format(country=country), cached)
        return cached
    except Exception as exc:  # the CRDDC bucket has answered 403 at times
        print(f"S3 no disponible para {country}: {exc}")
    with urllib.request.urlopen(FIGSHARE_FILES, timeout=60) as response:
        files = json.load(response)
    by_name = {item["name"]: item for item in files}
    if f"RDD2022_{country}.zip" in by_name:
        download(by_name[f"RDD2022_{country}.zip"]["download_url"], cached)
        return cached
    full = [item for item in files if item["name"].lower().endswith(".zip")]
    if not full:
        raise RuntimeError(
            f"No se encontró RDD2022 para {country}. Sube RDD2022_{country}.zip a {cache} y vuelve a ejecutar."
        )
    full_zip = work / full[0]["name"]
    if not full_zip.is_file():
        download(full[0]["download_url"], full_zip)
    return full_zip


def extract_country(country: str, archive: Path, raw: Path) -> Path:
    """Extract only the members of one country, unpacking a nested country ZIP if the archive has one."""
    target = raw / country
    if target.is_dir() and any(target.rglob("*.xml")):
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.namelist() if country.lower() in m.lower()]
        if not members:
            raise RuntimeError(f"{archive.name} no contiene archivos de {country}")
        bundle.extractall(target, members=members)
    for nested in list(target.rglob("*.zip")):
        with zipfile.ZipFile(nested) as inner:
            inner.extractall(nested.parent)
        nested.unlink()
    return target


def parse_annotation(xml_path: Path) -> list[str]:
    """YOLO lines for the four thesis classes, using the same rule as scripts/convert_voc_to_yolo.py."""
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width") or 0)
    height = float(root.findtext("size/height") or 0)
    if width <= 0 or height <= 0:
        from PIL import Image

        image = xml_path.parents[2] / "images" / f"{xml_path.stem}.jpg"
        width, height = Image.open(image).size
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_MAP:
            continue
        box = obj.find("bndbox")
        xmin = max(0.0, float(box.findtext("xmin")))
        ymin = max(0.0, float(box.findtext("ymin")))
        xmax = min(width, float(box.findtext("xmax")))
        ymax = min(height, float(box.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue
        lines.append(
            f"{CLASS_MAP[name]} {(xmin + xmax) / 2 / width:.6f} {(ymin + ymax) / 2 / height:.6f} "
            f"{(xmax - xmin) / width:.6f} {(ymax - ymin) / height:.6f}"
        )
    return lines


def labelled_images(country_dir: Path) -> list[tuple[Path, list[str]]]:
    """Only the train split of RDD2022 has labels; images without any of the four classes are dropped."""
    found = []
    for xml_dir in sorted(country_dir.rglob("annotations/xmls")):
        if xml_dir.parents[1].name != "train":
            continue
        images_dir = xml_dir.parents[1] / "images"
        for xml_path in sorted(xml_dir.glob("*.xml")):
            image = images_dir / f"{xml_path.stem}.jpg"
            lines = parse_annotation(xml_path)
            if image.is_file() and lines:
                found.append((image, lines))
    if not found:
        raise RuntimeError(f"Sin imágenes etiquetadas en {country_dir}")
    return found


def place(dataset: Path, split: str, image: Path, label_text: str, country: str) -> None:
    (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
    (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, dataset / "images" / split / image.name)
    (dataset / "labels" / split / f"{image.stem}.txt").write_text(label_text, encoding="utf-8")


def build_manifest(dataset: Path, countries: dict) -> dict:
    files, counts = [], {}
    for path in sorted(p for p in dataset.rglob("*") if p.is_file()):
        relative = path.relative_to(dataset)
        kind, split = relative.parts[0], relative.parts[1]
        files.append({"path": str(relative), "kind": kind, "split": split, "bytes": path.stat().st_size, "sha256": sha256(path)})
        if kind == "images":
            counts[split] = counts.get(split, 0) + 1
    return {"class_names": NAMES, "split_counts": counts, "countries": countries, "files": files}


def verify_manifest(dataset: Path, manifest: dict, full_hash: bool) -> dict:
    for item in manifest["files"]:
        path = dataset / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise FileNotFoundError(f"Falta o cambió: {path}")
        if full_hash and sha256(path) != item["sha256"]:
            raise AssertionError(f"SHA-256 distinto: {path}")
    return {"split_counts": manifest["split_counts"], "files_verified": len(manifest["files"]), "full_hash": full_hash}


def paths(results_root: Path) -> dict:
    return {
        "dataset": Path("/content/dataset_stage8"),
        "manifest": results_root / "dataset_stage8_manifest.json",
        "archive": results_root / "dataset_stage8.zip",
    }


def prepare(results_root: Path, cache: Path) -> None:
    cfg = load_config()
    p = paths(results_root)
    if p["manifest"].exists():
        raise FileExistsError("El dataset de la etapa 8 ya está preparado; usa la acción restore")
    stage6.verify_dataset(full_hash=True)
    dataset, work, raw = p["dataset"], Path("/content/rdd2022_work"), Path("/content/rdd2022_raw")
    if dataset.exists():
        shutil.rmtree(dataset)
    # China: the exact Stage-7 train and val files; its test split is not copied.
    for split, target in (("train", "train"), ("val", "val_china")):
        for image in sorted((CHINA_DATASET / "images" / split).glob("*.jpg")):
            label = CHINA_DATASET / "labels" / split / f"{image.stem}.txt"
            place(dataset, target, image, label.read_text(encoding="utf-8"), "China")
    countries = {}
    rng = random.Random(int(cfg["seed"]))
    for country in cfg["countries"]:
        country_dir = extract_country(country, fetch_country_zip(country, cache, work), raw)
        items = labelled_images(country_dir)
        rng.shuffle(items)
        chosen = items[: int(cfg["max_images_per_country"])]
        n_train = round(len(chosen) * float(cfg["split"]["train"]))
        n_val = round(len(chosen) * float(cfg["split"]["val"]))
        groups = {"train": chosen[:n_train], "val_paises": chosen[n_train:n_train + n_val], LOCKED_SPLIT: chosen[n_train + n_val:]}
        for split, group in groups.items():
            for image, lines in group:
                place(dataset, split, image, "\n".join(lines), country)
        countries[country] = {"labelled_available": len(items), "used": len(chosen), **{k: len(v) for k, v in groups.items()}}
        print(country, countries[country])
    manifest = build_manifest(dataset, countries)
    results_root.mkdir(parents=True, exist_ok=True)
    # The manifest goes last: its presence means the archive in Drive is complete.
    shutil.make_archive(str(p["archive"].with_suffix("")), "zip", dataset)
    p["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"split_counts": manifest["split_counts"], "countries": countries}, indent=2))


def restore(results_root: Path, full_hash: bool) -> dict:
    p = paths(results_root)
    manifest = json.loads(p["manifest"].read_text(encoding="utf-8"))
    if not p["dataset"].exists():
        with zipfile.ZipFile(p["archive"]) as bundle:
            bundle.extractall(p["dataset"])
    return verify_manifest(p["dataset"], manifest, full_hash)


def data_yaml(results_root: Path, val_splits: tuple[str, ...], name: str) -> Path:
    target = results_root / f"{name}.yaml"
    payload = {
        "path": str(paths(results_root)["dataset"]),
        "train": "images/train",
        "val": [f"images/{split}" for split in val_splits],
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def init_model(name: str) -> tuple[YOLO, dict]:
    cfg = load_config()
    spec = cfg["init_weights"][name]
    weights = Path(spec["drive_path"])
    if sha256(weights) != spec["sha256"]:
        raise AssertionError(f"El best.pt de la etapa 7 de {name} no coincide con el registrado")
    stage6.seed_everything(int(cfg["seed"]))
    model = YOLO(str(weights), task="detect", verbose=False)
    structure = audit_lmd(model) if name == "lmd_yolo" else audit_baseline(model)
    return model, {"init_weights": str(weights), "init_sha256": spec["sha256"], "structure": structure}


def train_kwargs(results_root: Path, cfg: dict) -> dict:
    kwargs = stage6.train_kwargs(data_yaml(results_root, ("val_china", "val_paises"), "data_stage8"), cfg)
    # Fine-tuning: keep the Stage-7 weights (stage6 trains from scratch with pretrained=False).
    kwargs["pretrained"] = True
    return kwargs


def evaluate(weights: Path, results_root: Path, out_dir: Path, cfg: dict) -> dict:
    """Validation metrics on all val images, on China only and on the other countries only."""
    report = {}
    for label, splits in (("val_total", ("val_china", "val_paises")), ("val_china", ("val_china",)), ("val_paises", ("val_paises",))):
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


def baseline(results_root: Path) -> None:
    """Stage-7 models on the Stage-8 validation sets: the 'before' column."""
    cfg = load_config()
    restore(results_root, full_hash=False)
    out = results_root / "baseline_stage7"
    if out.exists():
        raise FileExistsError(f"Ya existe {out}")
    report = {name: evaluate(Path(cfg["init_weights"][name]["drive_path"]), results_root, out / name, cfg) for name in ("yolov8n", "lmd_yolo")}
    (out / "baseline_stage7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def smoke(results_root: Path) -> None:
    cfg = load_config()
    dataset = restore(results_root, full_hash=True)
    smoke_root = results_root / "smoke"
    summaries = {}
    for name in ("yolov8n", "lmd_yolo"):
        if (smoke_root / name).exists():
            raise FileExistsError(f"Ya existe el smoke de {name}")
        model, _ = init_model(name)
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


def train_model(name: str, results_root: Path, resume: bool) -> None:
    cfg = load_config()
    gate_path = results_root / "smoke_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    if gate.get("status") != "PASS" or gate["config_sha256"] != sha256(CONFIG_PATH) or gate["dataset_manifest_sha256"] != sha256(paths(results_root)["manifest"]):
        raise RuntimeError("Falta un smoke válido de la etapa 8")
    restore(results_root, full_hash=False)
    project = results_root / "training"
    run_dir = project / name
    last_ckpt = run_dir / "weights" / "last.pt"
    resuming = resume and last_ckpt.is_file() and not (run_dir / "stage8_summary.json").exists()
    if run_dir.exists() and not resuming:
        raise FileExistsError(f"Ya existe la corrida de {name}; usa --resume si se cortó")
    model, provenance = init_model(name)
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
    seconds = float(rows[-1]["time"])
    summary = {
        "model": name, "status": "COMPLETE", "initialization": "stage7_best", **provenance,
        "config": CONFIG_PATH.name, "config_sha256": sha256(CONFIG_PATH), "resumed": resuming,
        "epochs": len(rows), "best_epoch": int(best_row["epoch"]), "training_seconds": seconds,
        "seconds_per_epoch": seconds / len(rows), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "validation": evaluate(best_pt, results_root, results_root / "best_validation" / name, cfg),
        "best_pt": str(best_pt), "best_pt_sha256": sha256(best_pt), "test_used": False,
    }
    (run_dir / "stage8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def compare(results_root: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    before = json.loads((results_root / "baseline_stage7" / "baseline_stage7.json").read_text(encoding="utf-8"))
    table, frames = {}, {}
    for name in ("yolov8n", "lmd_yolo"):
        run = results_root / "training" / name
        after = json.loads((run / "stage8_summary.json").read_text(encoding="utf-8"))
        frames[name] = pd.read_csv(run / "results.csv")
        table[name] = {
            split: {"stage7": before[name][split]["metrics"], "stage8": after["validation"][split]["metrics"]}
            for split in ("val_total", "val_china", "val_paises")
        }
        table[name]["summary"] = after
    columns = ["train/box_loss", "train/cls_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, column in zip(axes.flat, columns):
        for name, frame in frames.items():
            ax.plot(frame["epoch"], frame[column], label=name)
        ax.set_title(column); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out = results_root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "learning_curves.png", dpi=160)
    payload = {"status": "READY_FOR_TECHNICAL_REVIEW", "test_used": False, **table}
    (out / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name in ("yolov8n", "lmd_yolo"):
        for split in ("val_china", "val_paises", "val_total"):
            b, a = table[name][split]["stage7"]["metrics/mAP50(B)"], table[name][split]["stage8"]["metrics/mAP50(B)"]
            print(f"{name:9s} {split:10s} mAP@50 etapa 7 {b:.3f} -> etapa 8 {a:.3f}")


def main() -> None:
    global CONFIG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "restore", "baseline", "smoke", "train", "compare"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/content/drive/MyDrive/TP1_LMD/rdd2022"),
                        help="prepare: carpeta de Drive donde se guardan los ZIP de RDD2022 por país")
    parser.add_argument("--model", choices=("yolov8n", "lmd_yolo"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    CONFIG_PATH = args.config.expanduser().resolve()
    stage6.CONFIG_PATH = CONFIG_PATH
    results_root = args.results_root.expanduser().resolve()
    if args.action == "prepare":
        prepare(results_root, args.cache)
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
    else:
        compare(results_root)


if __name__ == "__main__":
    main()
