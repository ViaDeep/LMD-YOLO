"""Reproducible Stage-6 Colab workflow with an explicit smoke gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
sys.path.insert(0, str(VENDOR))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import DetectDBB, DiverseBranchBlock, MGConv  # noqa: E402
from ultralytics.utils.torch_utils import get_num_params  # noqa: E402


NAMES = ["D00", "D10", "D20", "D40"]
EXPECTED_COUNTS = {"train": 3082, "val": 385, "test": 386}
CONFIG_PATH = ROOT / "configs" / "comparison.yaml"
DATASET_MANIFEST = ROOT / "manifests" / "dataset_manifest.json"
PACKAGE_MANIFEST = ROOT / "manifests" / "package_manifest.json"
MODEL_YAMLS = {
    "yolov8n": ROOT / "models" / "yolov8n.yaml",
    "lmd_yolo": ROOT / "models" / "lmd_yolo_stage3.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def runtime_data_yaml(results_root: Path) -> Path:
    target = results_root / "runtime_data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str((ROOT / "dataset").resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {i: name for i, name in enumerate(NAMES)},
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def verify_dataset(full_hash: bool = True) -> dict:
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if manifest["class_names"] != NAMES or manifest["split_counts"] != EXPECTED_COUNTS:
        raise AssertionError("Dataset manifest metadata is not canonical")
    seen = {split: 0 for split in EXPECTED_COUNTS}
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise FileNotFoundError(f"Missing or size-mismatched dataset file: {path}")
        if full_hash and sha256(path) != item["sha256"]:
            raise AssertionError(f"Dataset SHA-256 mismatch: {path}")
        if item["kind"] == "image":
            seen[item["split"]] += 1
    if seen != EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected split counts: {seen}")
    return {"split_counts": seen, "files_verified": len(manifest["files"]), "full_hash": full_hash}


def build_model(name: str) -> YOLO:
    if name not in MODEL_YAMLS:
        raise ValueError(name)
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    return YOLO(str(MODEL_YAMLS[name]), task="detect", verbose=False)


def measure_gflops_cpu(network: torch.nn.Module, imgsz: int = 640) -> dict:
    """Profile a real CPU forward at full resolution; never convert errors to 0.0."""
    try:
        import thop
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics-thop is required for the Stage-6 FLOPs audit; "
            "install requirements-colab.txt before preflight"
        ) from exc

    profile_model = deepcopy(network).cpu().float().eval()
    sample = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32, device="cpu")
    try:
        macs, profiled_parameters = thop.profile(profile_model, inputs=(sample,), verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"THOP full-resolution CPU profiling failed for {type(network).__name__}; "
            "the failure is fatal rather than reported as 0.0"
        ) from exc
    del profile_model, sample
    parameters = get_num_params(network)
    if not math.isfinite(macs) or macs <= 0:
        raise AssertionError(f"THOP returned invalid MACs: {macs}")
    if int(profiled_parameters) != parameters:
        raise AssertionError(f"THOP parameter mismatch: {profiled_parameters} != {parameters}")
    return {
        "gflops": macs * 2.0 / 1e9,
        "macs": int(macs),
        "parameters": parameters,
        "method": "ultralytics-thop full 1x3x640x640 CPU forward; FLOPs=2*MACs",
        "thop_version": getattr(thop, "__version__", "unknown"),
    }


def audit_lmd(model: YOLO) -> dict:
    network = model.model.float()
    head = network.model[-1]
    dbbs = [m for m in network.modules() if isinstance(m, DiverseBranchBlock)]
    mg_positions = [i for i, layer in enumerate(network.model) if isinstance(layer, MGConv)]
    if type(head) is not DetectDBB or head.nc != 4 or head.reg_max != 16:
        raise AssertionError("LMD DetectDBB/nc/reg_max mismatch")
    if len(dbbs) != 6 or mg_positions != [21, 24]:
        raise AssertionError(f"LMD DBB/MGConv mismatch: {len(dbbs)}, {mg_positions}")
    if any(hasattr(block, "dbb_reparam") or not hasattr(block, "dbb_origin") for block in dbbs):
        raise AssertionError("LMD must contain six TRAIN multibranch DBBs")
    if any(network.model[i].groups != 2 for i in mg_positions):
        raise AssertionError("MGConv groups mismatch")
    strides = [int(x) for x in network.stride.tolist()]
    if strides != [8, 16, 32]:
        raise AssertionError(f"LMD stride mismatch: {strides}")
    profile = measure_gflops_cpu(network, imgsz=640)
    result = {
        "backbone": "LeYOLO-Nano", "head": "DetectDBB", "dbb_count": 6,
        "dbb_mode": "TRAIN multibranch", "mgconv_positions": mg_positions,
        "mgconv_groups": [2, 2], "classes": NAMES, "strides": strides,
        "parameters": get_num_params(network), "gflops_640": round(profile["gflops"], 6),
        "flops_profile": profile,
    }
    if result["parameters"] != 3_073_884 or result["gflops_640"] != 8.56617:
        raise AssertionError(f"LMD measurements changed: {result}")
    return result


def audit_baseline(model: YOLO) -> dict:
    network = model.model.float().eval()
    if network.model[-1].nc != 4 or [int(x) for x in network.stride.tolist()] != [8, 16, 32]:
        raise AssertionError("YOLOv8n baseline nc/strides mismatch")
    profile = measure_gflops_cpu(network, imgsz=640)
    return {
        "architecture": "YOLOv8n", "classes": NAMES,
        "strides": [int(x) for x in network.stride.tolist()],
        "parameters": get_num_params(network), "gflops_640": round(profile["gflops"], 6),
        "flops_profile": profile,
    }


def gpu_forward_backward(model: YOLO, name: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    network = model.model.float().cuda().train()
    x = torch.randn(2, 3, 640, 640, device="cuda")
    output = network(x)
    if not isinstance(output, dict) or not {"boxes", "scores"}.issubset(output):
        raise AssertionError(f"Unexpected {name} training output")
    loss = output["boxes"].float().square().mean() + output["scores"].float().square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite synthetic loss for {name}")
    loss.backward()
    peak = torch.cuda.max_memory_allocated()
    del x, output, loss, network
    torch.cuda.empty_cache()
    return {"forward": "PASS", "backward": "PASS", "peak_allocated_bytes_batch2": peak}


def preflight(results_root: Path, skip_full_hash: bool) -> None:
    import ultralytics

    if Path(ultralytics.__file__).resolve().parent != (VENDOR / "ultralytics").resolve():
        raise RuntimeError(f"Vanilla/wrong Ultralytics imported: {ultralytics.__file__}")
    if ultralytics.__version__ != "8.4.153":
        raise RuntimeError(f"Expected vendored Ultralytics 8.4.153, got {ultralytics.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Colab T4 GPU runtime")
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu.upper():
        raise RuntimeError(f"Expected Tesla T4, got {gpu}")
    dataset = verify_dataset(full_hash=not skip_full_hash)
    baseline = build_model("yolov8n")
    lmd = build_model("lmd_yolo")
    result = {
        "status": "PASS", "gpu": gpu,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": sys.version, "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
        "ultralytics": ultralytics.__version__, "ultralytics_path": ultralytics.__file__,
        "dataset": dataset, "baseline": audit_baseline(baseline), "lmd": audit_lmd(lmd),
        "gpu_checks": {
            "yolov8n": gpu_forward_backward(baseline, "yolov8n"),
            "lmd_yolo": gpu_forward_backward(lmd, "lmd_yolo"),
        },
        "config_sha256": sha256(CONFIG_PATH), "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
        "initialization": "scratch_both",
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def train_kwargs(data_yaml: Path, cfg: dict) -> dict:
    aug = cfg["augmentations"]
    return {
        "data": str(data_yaml), "imgsz": cfg["imgsz"], "batch": cfg["batch"],
        "device": 0, "workers": cfg["workers"], "optimizer": cfg["optimizer"],
        "lr0": cfg["lr0"], "lrf": cfg["lrf"], "momentum": cfg["momentum"],
        "weight_decay": cfg["weight_decay"], "warmup_epochs": cfg["warmup_epochs"],
        "warmup_momentum": cfg["warmup_momentum"], "warmup_bias_lr": cfg["warmup_bias_lr"],
        "cos_lr": cfg["cos_lr"], "seed": cfg["seed"], "deterministic": cfg["deterministic"],
        "patience": cfg["patience"], "amp": cfg["amp"], "pretrained": False,
        "val": True, "split": "val", "plots": True, "save": True, "verbose": True,
        **aug,
    }


def assert_finite_csv(path: Path) -> int:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        for key, value in row.items():
            if key != "epoch" and value and not math.isfinite(float(value)):
                raise FloatingPointError(f"Non-finite {key} in {path}")
    return len(rows)


def smoke(results_root: Path) -> None:
    preflight_path = results_root / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text())["status"] != "PASS":
        raise RuntimeError("Run a successful preflight first")
    cfg = load_config()
    data_yaml = runtime_data_yaml(results_root)
    smoke_root = results_root / "smoke"
    summaries = {}
    for name in ("yolov8n", "lmd_yolo"):
        run_dir = smoke_root / name
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite smoke run: {run_dir}")
        model = build_model(name)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        kwargs = train_kwargs(data_yaml, cfg)
        kwargs.update(
            epochs=cfg["smoke"]["epochs"], fraction=cfg["smoke"]["fraction"],
            batch=cfg["smoke"]["batch"], project=str(smoke_root), name=name, exist_ok=False,
        )
        model.train(**kwargs)
        rows = assert_finite_csv(run_dir / "results.csv")
        summaries[name] = {
            "status": "PASS", "rows": rows, "seconds": time.time() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "best_pt": str(run_dir / "weights" / "best.pt"),
            "last_pt": str(run_dir / "weights" / "last.pt"),
        }
    gate = {
        "status": "PASS", "models": summaries, "batch": cfg["batch"],
        "config_sha256": sha256(CONFIG_PATH), "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
        "message": "Smoke passed. Long training remains a separate manual action.",
    }
    (results_root / "smoke_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


def metric_dict(metrics) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.results_dict.items()}


def train_model(name: str, results_root: Path) -> None:
    gate_path = results_root / "smoke_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("No smoke gate. Run preflight and smoke, inspect them, then train manually.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["status"] != "PASS" or gate["config_sha256"] != sha256(CONFIG_PATH) or gate["dataset_manifest_sha256"] != sha256(DATASET_MANIFEST):
        raise RuntimeError("Smoke gate is stale or invalid")
    cfg = load_config()
    data_yaml = runtime_data_yaml(results_root)
    project = results_root / "training"
    run_dir = project / name
    val_dir = results_root / "best_validation" / name
    if run_dir.exists() or val_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Stage-6 results for {name}")
    model = build_model(name)
    structure = audit_baseline(model) if name == "yolov8n" else audit_lmd(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    returned = model.train(
        **train_kwargs(data_yaml, cfg), epochs=cfg["epochs"],
        project=str(project), name=name, exist_ok=False, save_period=5,
    )
    training_seconds = time.time() - started
    peak_memory = torch.cuda.max_memory_allocated()
    csv_path = run_dir / "results.csv"
    row_count = assert_finite_csv(csv_path)
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    best_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    best_pt, last_pt = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    best_model = YOLO(str(best_pt), task="detect", verbose=False)
    if name == "lmd_yolo":
        audit_lmd(best_model)
    val = best_model.val(
        data=str(data_yaml), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"],
        device=0, workers=cfg["workers"], project=str(val_dir.parent), name=name,
        exist_ok=False, plots=True, verbose=True,
    )
    per_class = {
        class_name: {
            "precision": float(val.box.p[i]), "recall": float(val.box.r[i]),
            "mAP50": float(val.box.ap50[i]), "mAP50_95": float(val.box.maps[i]),
        }
        for i, class_name in enumerate(NAMES)
    }
    summary = {
        "model": name, "status": "COMPLETE", "initialization": "scratch",
        "structure": structure, "epochs": row_count, "best_epoch": int(best_row["epoch"]),
        "training_seconds": training_seconds, "seconds_per_epoch": training_seconds / row_count,
        "peak_allocated_bytes": peak_memory, "training_return_metrics": metric_dict(returned),
        "explicit_best_validation": metric_dict(val), "per_class": per_class,
        "best_pt": str(best_pt), "last_pt": str(last_pt), "results_csv": str(csv_path),
        "best_pt_sha256": sha256(best_pt), "last_pt_sha256": sha256(last_pt),
    }
    (run_dir / "stage6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def compare(results_root: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    summaries, frames = {}, {}
    for name in ("yolov8n", "lmd_yolo"):
        run = results_root / "training" / name
        summaries[name] = json.loads((run / "stage6_summary.json").read_text(encoding="utf-8"))
        frames[name] = pd.read_csv(run / "results.csv")
    columns = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, column in zip(axes.flat, columns):
        for name, frame in frames.items():
            ax.plot(frame["epoch"], frame[column], label=name)
        ax.set_title(column); ax.grid(True, alpha=0.3); ax.legend()
    axes.flat[-1].axis("off")
    fig.tight_layout()
    comparison_dir = results_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(comparison_dir / "learning_curves.png", dpi=160)
    payload = {
        "status": "READY_FOR_TECHNICAL_REVIEW", "test_used": False,
        "yolov8n": summaries["yolov8n"], "lmd_yolo": summaries["lmd_yolo"],
        "note": "Do not declare a winner from one metric; review stability, convergence, classes, time and compute cost.",
    }
    (comparison_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "smoke", "train", "compare"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model", choices=("yolov8n", "lmd_yolo"))
    parser.add_argument("--skip-full-hash", action="store_true", help="Size/count check only; full hash is recommended")
    args = parser.parse_args()
    results_root = args.results_root.expanduser().resolve()
    if args.action == "preflight":
        preflight(results_root, args.skip_full_hash)
    elif args.action == "smoke":
        smoke(results_root)
    elif args.action == "train":
        if not args.model:
            parser.error("train requires --model")
        train_model(args.model, results_root)
    else:
        compare(results_root)


if __name__ == "__main__":
    main()
