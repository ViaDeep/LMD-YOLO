"""Build and verify Stage-6 dataset/package SHA-256 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"
SPLITS = {"train": 3082, "val": 385, "test": 386}
NAMES = ["D00", "D10", "D20", "D40"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label(path: Path) -> Counter:
    counts = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        values = list(map(float, fields))
        cls = int(values[0])
        if values[0] != cls or cls not in range(4):
            raise ValueError(f"{path}:{line_number}: class outside 0-3")
        x, y, w, h = values[1:]
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_number}: invalid normalized coordinates")
        counts[cls] += 1
    return counts


def build_dataset_manifest() -> dict:
    files, objects = [], Counter()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split, expected in SPLITS.items():
        image_dir = ROOT / "dataset" / "images" / split
        label_dir = ROOT / "dataset" / "labels" / split
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions)
        labels = sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        if len(images) != expected or len(labels) != expected:
            raise AssertionError(f"{split}: expected {expected}/{expected}, got {len(images)}/{len(labels)}")
        image_stems, label_stems = {p.stem for p in images}, {p.stem for p in labels}
        if image_stems != label_stems:
            raise AssertionError(f"{split}: image/label stem mismatch")
        for image in images:
            files.append({
                "path": image.relative_to(ROOT).as_posix(), "split": split, "kind": "image",
                "bytes": image.stat().st_size, "sha256": sha256(image),
            })
        for label in labels:
            objects.update(validate_label(label))
            files.append({
                "path": label.relative_to(ROOT).as_posix(), "split": split, "kind": "label",
                "bytes": label.stat().st_size, "sha256": sha256(label),
            })
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item['path']}|{item['bytes']}|{item['sha256']}\n".encode())
    result = {
        "schema": 1, "class_names": NAMES, "split_counts": SPLITS,
        "object_counts_all_splits": {NAMES[i]: objects[i] for i in range(4)},
        "files": files, "combined_sha256": digest.hexdigest(),
    }
    MANIFEST_DIR.mkdir(exist_ok=True)
    (MANIFEST_DIR / "dataset_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def package_files() -> list[Path]:
    excluded = {"manifests/package_manifest.json", "colab_stage6.zip"}
    return sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.relative_to(ROOT).as_posix() not in excluded and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )


def build_package_manifest() -> dict:
    files = [
        {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in package_files()
    ]
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item['path']}|{item['bytes']}|{item['sha256']}\n".encode())
    result = {"schema": 1, "file_count": len(files), "files": files, "combined_sha256": digest.hexdigest()}
    MANIFEST_DIR.mkdir(exist_ok=True)
    (MANIFEST_DIR / "package_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify_package() -> dict:
    path = MANIFEST_DIR / "package_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checked = 0
    for item in manifest["files"]:
        target = ROOT / item["path"]
        if not target.is_file() or target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            raise AssertionError(f"Package mismatch: {target}")
        checked += 1
    return {"status": "PASS", "files_verified": checked, "combined_sha256": manifest["combined_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build-dataset", "build-package", "verify-package"))
    args = parser.parse_args()
    if args.action == "build-dataset":
        result = build_dataset_manifest()
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    elif args.action == "build-package":
        result = build_package_manifest()
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    else:
        print(json.dumps(verify_package(), indent=2))


if __name__ == "__main__":
    main()
