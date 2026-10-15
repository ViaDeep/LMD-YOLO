"""Generate canonical mobile-preprocessing golden assets without changing the ONNX.

The canonical resizer is deliberately implemented here instead of delegating to
OpenCV.  The matching Kotlin implementation uses the same integer equations,
which makes letterbox -> RGB -> NCHW reproducible once the decoded RGB8 pixels
are fixed.  JPEG decoding is measured separately because different JPEG
decoders are not required to produce identical RGB8 samples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/stage4_mobile_export_audit/onnx_fp32_640/best_deploy_fp32_source_copy.onnx"
IMAGE = ROOT / "data/dataset/images/val/China_Drone_000001.jpg"
ASSETS = ROOT / "mobile_benchmark/app/src/main/assets"
EXPECTED_MODEL_SHA256 = "a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8"
TARGET_H = 640
TARGET_W = 640
PADDING_VALUE = 114
CLASSES = ["D00", "D10", "D20", "D40"]
CONFIDENCE = 0.25
IOU = 0.45


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_divide_ties_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice < denominator:
        return quotient
    if twice > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def letterbox_geometry(src_h: int, src_w: int) -> dict[str, int | float]:
    # Equivalent geometry to Ultralytics LetterBox(new_shape=(640, 640),
    # auto=False, scale_fill=False, scaleup=True, center=True).
    if TARGET_H * src_w <= TARGET_W * src_h:  # height limits the scale
        resized_h = TARGET_H
        resized_w = round_divide_ties_to_even(src_w * TARGET_H, src_h)
    else:  # width limits the scale
        resized_w = TARGET_W
        resized_h = round_divide_ties_to_even(src_h * TARGET_W, src_w)
    pad_w = TARGET_W - resized_w
    pad_h = TARGET_H - resized_h
    left = pad_w // 2
    top = pad_h // 2
    return {
        "source_height": src_h,
        "source_width": src_w,
        "target_height": TARGET_H,
        "target_width": TARGET_W,
        "resized_height": resized_h,
        "resized_width": resized_w,
        "pad_left": left,
        "pad_right": pad_w - left,
        "pad_top": top,
        "pad_bottom": pad_h - top,
        "scale": min(TARGET_H / src_h, TARGET_W / src_w),
    }


def axis_map(src_len: int, dst_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return exact half-pixel bilinear indices and integer weights."""
    denominator = 2 * dst_len
    dst = np.arange(dst_len, dtype=np.int64)
    position_numerator = (2 * dst + 1) * src_len - dst_len
    lo = np.floor_divide(position_numerator, denominator)
    remainder = position_numerator - lo * denominator
    hi = lo + 1

    below = position_numerator <= 0
    above = position_numerator >= (src_len - 1) * denominator
    lo = np.where(below, 0, np.where(above, src_len - 1, lo))
    hi = np.where(below, 0, np.where(above, src_len - 1, hi))
    weight_hi = np.where(below | above, 0, remainder)
    weight_lo = denominator - weight_hi
    return lo, hi, weight_lo, weight_hi, denominator


def resize_bilinear_canonical(rgb: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    src_h, src_w, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 HWC input")
    x0, x1, wx0, wx1, den_x = axis_map(src_w, dst_w)
    y0, y1, wy0, wy1, den_y = axis_map(src_h, dst_h)
    denominator = den_x * den_y
    output = np.empty((dst_h, dst_w, 3), dtype=np.uint8)
    source = rgb.astype(np.int64, copy=False)
    for y in range(dst_h):
        top = source[y0[y]]
        bottom = source[y1[y]]
        numerator = (
            top[x0] * (wx0 * wy0[y])[:, None]
            + top[x1] * (wx1 * wy0[y])[:, None]
            + bottom[x0] * (wx0 * wy1[y])[:, None]
            + bottom[x1] * (wx1 * wy1[y])[:, None]
        )
        # Explicit round-half-up. All terms are non-negative integers.
        output[y] = ((numerator + denominator // 2) // denominator).astype(np.uint8)
    return output


def canonical_letterbox(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
    geometry = letterbox_geometry(rgb.shape[0], rgb.shape[1])
    resized = resize_bilinear_canonical(
        rgb, int(geometry["resized_height"]), int(geometry["resized_width"])
    )
    canvas = np.full((TARGET_H, TARGET_W, 3), PADDING_VALUE, dtype=np.uint8)
    top = int(geometry["pad_top"])
    left = int(geometry["pad_left"])
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas, geometry


def compare_u8(left: np.ndarray, right: np.ndarray) -> dict[str, int | float]:
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {
        "elements": int(delta.size),
        "different_elements": int(np.count_nonzero(delta)),
        "max_abs_u8": int(delta.max()),
        "mean_abs_u8": float(delta.mean()),
    }


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
        candidates.append({
            "class_id": class_id,
            "class_name": CLASSES[class_id],
            "score": score,
            "xyxy": [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        })
    kept: list[dict] = []
    for class_id in range(len(CLASSES)):
        pending = sorted(
            (item for item in candidates if item["class_id"] == class_id),
            key=lambda item: item["score"], reverse=True,
        )
        while pending:
            best = pending.pop(0)
            kept.append(best)
            pending = [item for item in pending if iou(best, item) <= IOU]
    return sorted(kept, key=lambda item: item["score"], reverse=True), len(candidates)


def main() -> None:
    if sha256_file(MODEL) != EXPECTED_MODEL_SHA256:
        raise AssertionError("Frozen ONNX hash mismatch")

    # Ignore EXIF orientation explicitly. Orientation normalization is an
    # upstream production contract, not an implicit decoder side effect.
    bgr = cv2.imread(str(IMAGE), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise FileNotFoundError(IMAGE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    letterboxed, geometry = canonical_letterbox(rgb)
    input_tensor = np.ascontiguousarray(
        letterboxed.transpose(2, 0, 1)[None], dtype=np.float32
    ) / np.float32(255.0)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    output = np.asarray(
        session.run(None, {session.get_inputs()[0].name: input_tensor})[0], dtype=np.float32
    )
    if input_tensor.shape != (1, 3, TARGET_H, TARGET_W) or output.shape != (1, 8, 8400):
        raise AssertionError(f"Unexpected shapes: {input_tensor.shape}, {output.shape}")
    detections, candidate_count = postprocess(output)

    cv_linear = cv2.resize(rgb, (int(geometry["resized_width"]), int(geometry["resized_height"])), interpolation=cv2.INTER_LINEAR)
    cv_exact = cv2.resize(rgb, (int(geometry["resized_width"]), int(geometry["resized_height"])), interpolation=cv2.INTER_LINEAR_EXACT)
    resized = letterboxed[
        int(geometry["pad_top"]): int(geometry["pad_top"]) + int(geometry["resized_height"]),
        int(geometry["pad_left"]): int(geometry["pad_left"]) + int(geometry["resized_width"]),
    ]

    source_bytes = np.ascontiguousarray(rgb).tobytes()
    input_bytes = input_tensor.astype("<f4", copy=False).tobytes()
    output_bytes = output.astype("<f4", copy=False).tobytes()
    (ASSETS / "canonical_source_rgb_u8.bin").write_bytes(source_bytes)
    (ASSETS / "canonical_input_fp32.bin").write_bytes(input_bytes)
    (ASSETS / "canonical_output_fp32.bin").write_bytes(output_bytes)

    metadata = {
        "schema_version": 1,
        "model_sha256": sha256_file(MODEL),
        "image_name": IMAGE.name,
        "image_sha256": sha256_file(IMAGE),
        "pc_decoder": "OpenCV IMREAD_COLOR|IMREAD_IGNORE_ORIENTATION, BGR->RGB8",
        "orientation_contract": "Input pixel matrix is already upright; decoders must not apply EXIF implicitly",
        "color_contract": "opaque sRGB RGB8",
        "source_rgb_shape": list(rgb.shape),
        "source_rgb_sha256": sha256_bytes(source_bytes),
        "letterbox": geometry | {
            "auto": False,
            "scale_fill": False,
            "scaleup": True,
            "center": True,
            "padding_value_rgb": [PADDING_VALUE] * 3,
            "interpolation": "canonical half-pixel bilinear, exact rational integer weights, round-half-up",
        },
        "tensor": {
            "shape": list(input_tensor.shape),
            "layout": "NCHW",
            "dtype": "float32",
            "channel_order": "RGB",
            "normalization": "uint8 / float32(255.0)",
            "sha256": sha256_bytes(input_bytes),
        },
        "output": {
            "shape": list(output.shape),
            "dtype": "float32",
            "sha256": sha256_bytes(output_bytes),
            "onnxruntime_version": ort.__version__,
            "execution_provider": "CPUExecutionProvider",
        },
        "resizer_comparison_pc": {
            "canonical_vs_opencv_inter_linear": compare_u8(resized, cv_linear),
            "canonical_vs_opencv_inter_linear_exact": compare_u8(resized, cv_exact),
        },
        "confidence_threshold": CONFIDENCE,
        "iou_threshold": IOU,
        "candidate_count": candidate_count,
        "detections": detections,
    }
    (ASSETS / "canonical_preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
