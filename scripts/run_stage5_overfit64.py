"""Stage 5: reproducible controlled overfit of canonical TRAIN LMD_YOLO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import DetectDBB, DiverseBranchBlock, MGConv  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


SEED = 20260923
NAMES = ["D00", "D10", "D20", "D40"]
SOURCE_IMAGES = ROOT / "data" / "dataset" / "images" / "train"
SOURCE_LABELS = ROOT / "data" / "dataset" / "labels" / "train"
SUBSET = ROOT / "data" / "overfit64"
OUTPUT = ROOT / "outputs" / "stage5_overfit64"
RUNS = ROOT / "runs" / "detect"
RUN_NAME = "lmd_stage5_overfit64"
VAL_NAME = "lmd_stage5_overfit64_best_val"
SOURCE_YAML = ROOT / "models" / "yolov8n_leyolo_nano_mgconv_detect_dbb_stage3.yaml"
SOURCE_CKPT = ROOT / "runs" / "detect" / "leyolo_nano_mgconv_dbb_stage3_smoke" / "weights" / "best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        cls_float, x, y, w, h = map(float, fields)
        cls = int(cls_float)
        if cls_float != cls or cls not in range(4):
            raise ValueError(f"{path}:{line_number}: invalid class {cls_float}")
        values = (x, y, w, h)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}:{line_number}: NaN/Inf coordinate")
        if not all(0.0 <= v <= 1.0 for v in values) or w <= 0.0 or h <= 0.0:
            raise ValueError(f"{path}:{line_number}: invalid normalized box {values}")
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box extends outside image {values}")
        rows.append((cls, x, y, w, h))
    return rows


def image_records() -> list[dict]:
    records = []
    for image in sorted(p for p in SOURCE_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        label = SOURCE_LABELS / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Training image lacks label: {image}")
        boxes = read_label(label)
        if boxes:
            records.append({"image": image, "label": label, "boxes": boxes, "classes": {b[0] for b in boxes}})
    return records


def select_records(records: list[dict]) -> list[dict]:
    """Fixed stratified image quotas, deliberately oversampling minority D40."""
    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)
    selected: list[dict] = []
    used: set[str] = set()
    for cls, quota in ((3, 24), (2, 16), (1, 12), (0, 12)):
        candidates = [r for r in shuffled if cls in r["classes"] and r["image"].name not in used]
        if len(candidates) < quota:
            raise RuntimeError(f"Only {len(candidates)} unused train images available for class {cls}, need {quota}")
        for record in candidates[:quota]:
            selected.append(record)
            used.add(record["image"].name)
    if len(selected) != 64 or len(used) != 64:
        raise AssertionError("Selection is not exactly 64 unique images")
    return selected


COLORS = [(255, 90, 40), (30, 190, 60), (30, 130, 255), (200, 60, 220)]


def draw_gt(image_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    height, width = image.shape[:2]
    for cls, x, y, w, h in boxes:
        x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
        x2, y2 = int((x + w / 2) * width), int((y + h / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[cls], 2)
        cv2.putText(image, NAMES[cls], (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cls], 2)
    return image


def fit_tile(image: np.ndarray, size: tuple[int, int] = (480, 320)) -> np.ndarray:
    target_w, target_h = size
    scale = min(target_w / image.shape[1], target_h / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_mosaic(images: list[np.ndarray], labels: list[str], path: Path, columns: int = 4) -> None:
    tiles = []
    for image, label in zip(images, labels):
        tile = fit_tile(image)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    blank = np.full_like(tiles[0], 245)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [np.concatenate(tiles[i : i + columns], axis=1) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def audit_model(model: YOLO) -> dict:
    network = model.model.float()
    head = network.model[-1]
    dbbs = [m for m in network.modules() if isinstance(m, DiverseBranchBlock)]
    mg_positions = [i for i, layer in enumerate(network.model) if isinstance(layer, MGConv)]
    if type(head) is not DetectDBB or head.nc != 4 or head.reg_max != 16:
        raise AssertionError("Checkpoint does not contain canonical DetectDBB/nc/reg_max")
    if len(dbbs) != 6 or mg_positions != [21, 24]:
        raise AssertionError(f"Checkpoint structure changed: dbbs={len(dbbs)}, MGConv={mg_positions}")
    if any(hasattr(block, "dbb_reparam") for block in dbbs):
        raise AssertionError("A DBB is reparameterized; TRAIN multibranch is required")
    if [int(s) for s in network.stride.tolist()] != [8, 16, 32]:
        raise AssertionError(f"Unexpected strides: {network.stride.tolist()}")
    if any(network.model[i].groups != 2 for i in mg_positions):
        raise AssertionError("MGConv groups changed")
    return {
        "backbone": "LeYOLO-Nano",
        "mgconv_positions": mg_positions,
        "mgconv_groups": [network.model[i].groups for i in mg_positions],
        "head": type(head).__name__,
        "dbb_count": len(dbbs),
        "dbb_mode": "TRAIN multibranch",
        "classes": NAMES,
        "strides": [int(s) for s in network.stride.tolist()],
        "parameters": get_num_params(network),
        "gflops_640": round(get_flops(network.eval(), imgsz=640), 6),
    }


def prepare() -> None:
    for target in (SUBSET, OUTPUT, RUNS / RUN_NAME, RUNS / VAL_NAME):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite Stage-5 artifact: {target}")
    if not SOURCE_CKPT.is_file() or not SOURCE_YAML.is_file():
        raise FileNotFoundError("Canonical Stage-3 YAML/checkpoint is missing")
    records = select_records(image_records())
    (SUBSET / "images" / "train").mkdir(parents=True)
    (SUBSET / "labels" / "train").mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    class_counts = Counter()
    image_class_counts = Counter()
    manifest = []
    for index, record in enumerate(records, 1):
        image_dst = SUBSET / "images" / "train" / record["image"].name
        label_dst = SUBSET / "labels" / "train" / record["label"].name
        shutil.copy2(record["image"], image_dst)
        shutil.copy2(record["label"], label_dst)
        for cls, *_ in record["boxes"]:
            class_counts[cls] += 1
        for cls in record["classes"]:
            image_class_counts[cls] += 1
        manifest.append({
            "index": index,
            "image": record["image"].name,
            "source_image": str(record["image"]),
            "source_label": str(record["label"]),
            "image_sha256": sha256(record["image"]),
            "label_sha256": sha256(record["label"]),
            "objects": {NAMES[c]: sum(1 for b in record["boxes"] if b[0] == c) for c in range(4)},
        })
    data_yaml = {
        "path": str(SUBSET).replace("\\", "/"),
        "train": "images/train",
        "val": "images/train",
        "test": "images/train",
        "names": {i: name for i, name in enumerate(NAMES)},
    }
    (SUBSET / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "image", "source_image", "source_label", "image_sha256", "label_sha256", *NAMES])
        writer.writeheader()
        for item in manifest:
            writer.writerow({**{k: item[k] for k in writer.fieldnames if k not in NAMES}, **item["objects"]})
    visual_records = []
    visual_used = set()
    for cls in range(4):
        for record in records:
            if cls in record["classes"] and record["image"].name not in visual_used:
                visual_records.append(record)
                visual_used.add(record["image"].name)
                break
    for record in records:
        if len(visual_records) >= 12:
            break
        if record["image"].name not in visual_used:
            visual_records.append(record)
            visual_used.add(record["image"].name)
    make_mosaic(
        [draw_gt(r["image"], r["boxes"]) for r in visual_records],
        [r["image"].name for r in visual_records],
        OUTPUT / "labels_mosaic.jpg",
    )
    model = YOLO(str(SOURCE_CKPT), task="detect", verbose=False)
    structure = audit_model(model)
    summary = {
        "seed": SEED,
        "selection": "24 unique images containing D40, then 16 D20, 12 D10, 12 D00; seeded shuffle without replacement",
        "source_split": "train only",
        "train_and_evaluation_are_identical": True,
        "images": len(records),
        "objects_per_class": {NAMES[c]: class_counts[c] for c in range(4)},
        "images_containing_class": {NAMES[c]: image_class_counts[c] for c in range(4)},
        "all_images_exist": all((SUBSET / "images" / "train" / r["image"].name).is_file() for r in records),
        "all_labels_exist": all((SUBSET / "labels" / "train" / r["label"].name).is_file() for r in records),
        "labels_valid": True,
        "source_yaml": str(SOURCE_YAML),
        "source_checkpoint": str(SOURCE_CKPT),
        "source_checkpoint_sha256": sha256(SOURCE_CKPT),
        "structure": structure,
        "fixed_visual_sample": [r["image"].name for r in visual_records],
    }
    (OUTPUT / "preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("STAGE5_PREFLIGHT=" + json.dumps(summary, sort_keys=True))


def metric_dict(metrics) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.results_dict.items()}


def train_and_analyze() -> None:
    if not (OUTPUT / "preflight.json").is_file():
        raise RuntimeError("Run --prepare before training")
    for target in (RUNS / RUN_NAME, RUNS / VAL_NAME):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite Stage-5 run: {target}")
    started = time.time()
    model = YOLO(str(SOURCE_CKPT), task="detect", verbose=False)
    before = audit_model(model)
    metrics = model.train(
        data=str(SUBSET / "data.yaml"), epochs=50, imgsz=640, batch=8,
        device="cpu", workers=0, project=str(RUNS), name=RUN_NAME, exist_ok=False,
        optimizer="AdamW", lr0=0.001, lrf=0.1, momentum=0.9, weight_decay=0.0005,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.01,
        seed=SEED, deterministic=True, patience=100, save=True, save_period=10,
        pretrained=False, amp=False, val=True, plots=True, verbose=True,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
        scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
        bgr=0.0, multi_scale=0.0, close_mosaic=0,
    )
    training_seconds = time.time() - started
    best = RUNS / RUN_NAME / "weights" / "best.pt"
    last = RUNS / RUN_NAME / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("Training completed without best.pt/last.pt")
    best_model = YOLO(str(best), task="detect", verbose=False)
    after = audit_model(best_model)
    val_started = time.time()
    val = best_model.val(
        data=str(SUBSET / "data.yaml"), split="val", imgsz=640, batch=8,
        device="cpu", workers=0, project=str(RUNS), name=VAL_NAME,
        exist_ok=False, plots=True, verbose=True,
    )
    val_seconds = time.time() - val_started
    box = val.box
    per_class = {}
    for cls, name in enumerate(NAMES):
        per_class[name] = {
            "precision": float(box.p[cls]), "recall": float(box.r[cls]),
            "mAP50": float(box.ap50[cls]), "mAP50_95": float(box.maps[cls]),
        }
    visual_names = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))["fixed_visual_sample"][:8]
    comparison_dir = OUTPUT / "predictions"
    comparison_dir.mkdir()
    rendered = []
    prediction_results = best_model.predict(
        source=[str(SUBSET / "images" / "train" / name) for name in visual_names],
        imgsz=640, conf=0.05, iou=0.7, device="cpu", verbose=False,
    )
    for name, result in zip(visual_names, prediction_results):
        label_path = SUBSET / "labels" / "train" / f"{Path(name).stem}.txt"
        gt = draw_gt(SUBSET / "images" / "train" / name, read_label(label_path))
        pred = result.plot(labels=True, conf=True)
        gt_tile, pred_tile = fit_tile(gt), fit_tile(pred)
        cv2.putText(gt_tile, "GROUND TRUTH", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        cv2.putText(pred_tile, "PREDICTION best.pt", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        side = np.concatenate([gt_tile, pred_tile], axis=1)
        path = comparison_dir / f"{Path(name).stem}_gt_vs_pred.jpg"
        cv2.imwrite(str(path), side)
        rendered.append(side)
    make_mosaic(rendered, visual_names, OUTPUT / "predictions_mosaic.jpg", columns=2)
    summary = {
        "training_seconds": training_seconds,
        "explicit_best_validation_seconds": val_seconds,
        "total_seconds": time.time() - started,
        "initial_structure": before,
        "best_structure": after,
        "training_return_metrics": metric_dict(metrics),
        "best_validation_metrics": metric_dict(val),
        "per_class": per_class,
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256(best),
        "last_checkpoint": str(last),
        "results_csv": str(RUNS / RUN_NAME / "results.csv"),
        "prediction_images": [str(comparison_dir / f"{Path(name).stem}_gt_vs_pred.jpg") for name in visual_names],
        "configuration": {
            "epochs": 50, "imgsz": 640, "batch": 8, "seed": SEED,
            "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.1,
            "beta1_momentum_argument": 0.9, "weight_decay": 0.0005,
            "warmup_epochs": 3.0, "patience": 100,
            "augmentations": "all geometric/photometric transforms, mosaic, mixup, cutmix and copy-paste disabled",
            "train_split": "same 64 images", "validation_split": "same 64 images",
        },
    }
    (OUTPUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("STAGE5_SUMMARY=" + json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        train_and_analyze()


if __name__ == "__main__":
    main()
