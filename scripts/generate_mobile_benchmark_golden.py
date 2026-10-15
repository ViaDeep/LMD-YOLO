"""Generate immutable PC golden assets for the Android ONNX benchmark harness."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/stage4_mobile_export_audit/onnx_fp32_640/best_deploy_fp32_source_copy.onnx"
IMAGE = ROOT / "data/dataset/images/val/China_Drone_000001.jpg"
ASSETS = ROOT / "mobile_benchmark/app/src/main/assets"
EXPECTED_MODEL_SHA256 = "a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8"
CLASSES = ["D00", "D10", "D20", "D40"]
CONFIDENCE = 0.25
IOU = 0.45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(a: dict, b: dict) -> float:
    x1, y1 = max(a["xyxy"][0], b["xyxy"][0]), max(a["xyxy"][1], b["xyxy"][1])
    x2, y2 = min(a["xyxy"][2], b["xyxy"][2]), min(a["xyxy"][3], b["xyxy"][3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["xyxy"][2] - a["xyxy"][0]) * max(0.0, a["xyxy"][3] - a["xyxy"][1])
    area_b = max(0.0, b["xyxy"][2] - b["xyxy"][0]) * max(0.0, b["xyxy"][3] - b["xyxy"][1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def postprocess(output: np.ndarray) -> tuple[list[dict], int]:
    prediction = output[0]
    candidates: list[dict] = []
    for anchor in range(prediction.shape[1]):
        scores = prediction[4:, anchor]
        class_id = int(scores.argmax())
        score = float(scores[class_id])
        if score < CONFIDENCE:
            continue
        cx, cy, width, height = (float(v) for v in prediction[:4, anchor])
        candidates.append(
            {
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "score": score,
                "xyxy": [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            }
        )
    kept: list[dict] = []
    for class_id in range(len(CLASSES)):
        pending = sorted((item for item in candidates if item["class_id"] == class_id), key=lambda x: x["score"], reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            pending = [item for item in pending if iou(best, item) <= IOU]
    return sorted(kept, key=lambda x: x["score"], reverse=True), len(candidates)


def main() -> None:
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        raise AssertionError("Frozen ONNX hash mismatch")
    if ASSETS.exists():
        raise FileExistsError(f"Refusing to overwrite existing assets directory: {ASSETS}")
    ASSETS.mkdir(parents=True, exist_ok=False)

    bgr = cv2.imread(str(IMAGE), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(IMAGE)
    rgb = cv2.cvtColor(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    input_tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.0)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    output = np.asarray(session.run(None, {session.get_inputs()[0].name: input_tensor})[0], dtype=np.float32)
    if input_tensor.shape != (1, 3, 640, 640) or output.shape != (1, 8, 8400):
        raise AssertionError(f"Unexpected shapes: {input_tensor.shape}, {output.shape}")
    detections, candidate_count = postprocess(output)
    score_plane = output[0, 4:, :]
    max_score_flat_index = int(score_plane.argmax())
    max_score_class, max_score_anchor = np.unravel_index(max_score_flat_index, score_plane.shape)

    shutil.copy2(MODEL, ASSETS / MODEL.name)
    shutil.copy2(IMAGE, ASSETS / IMAGE.name)
    input_tensor.astype("<f4", copy=False).tofile(ASSETS / "golden_input_fp32.bin")
    output.astype("<f4", copy=False).tofile(ASSETS / "golden_output_fp32.bin")
    representative_indices = [0, 1, 100, 8399, 8400, 16800, 25200, 33600, 42000, 50400, 58800, 67199]
    metadata = {
        "model_sha256": sha256(MODEL),
        "model_bytes": MODEL.stat().st_size,
        "image_name": IMAGE.name,
        "image_sha256": sha256(IMAGE),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "input_sha256": hashlib.sha256(input_tensor.astype("<f4", copy=False).tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.astype("<f4", copy=False).tobytes()).hexdigest(),
        "onnxruntime_version": ort.__version__,
        "execution_provider": "CPUExecutionProvider",
        "preprocessing": "OpenCV imread BGR, resize 640x640 INTER_LINEAR, BGR->RGB, NCHW FP32 /255",
        "confidence_threshold": CONFIDENCE,
        "iou_threshold": IOU,
        "class_names": CLASSES,
        "candidate_count": candidate_count,
        "max_class_score": float(score_plane[max_score_class, max_score_anchor]),
        "max_class_score_class_id": int(max_score_class),
        "max_class_score_anchor": int(max_score_anchor),
        "detections": detections,
        "representative_output_values": [{"index": i, "value": float(output.reshape(-1)[i])} for i in representative_indices],
    }
    (ASSETS / "golden_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"assets": str(ASSETS), "model_sha256": metadata["model_sha256"], "detections": len(detections)}, indent=2))


if __name__ == "__main__":
    main()
