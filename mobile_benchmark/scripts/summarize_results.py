"""Summarize the latest pulled CPU/XNNPACK and NNAPI Android reports."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def latest(prefix: str) -> dict | None:
    paths = sorted(RESULTS.glob(f"{prefix}_*.json"), key=lambda item: item.stat().st_mtime)
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None


def nested(report: dict | None, *keys, default="—"):
    value = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main() -> None:
    cpu, nnapi = latest("cpu_xnnpack"), latest("nnapi")
    rows = [
        ("Estado", ("status",), ("status",)),
        ("Carga sesión (ms)", ("session_load_ms",), ("session_load_ms",)),
        ("Preprocesamiento medio (ms)", ("summary", "preprocessing_ms", "mean"), ("summary", "preprocessing_ms", "mean")),
        ("Inferencia media (ms)", ("summary", "inference_ms", "mean"), ("summary", "inference_ms", "mean")),
        ("p50 (ms)", ("summary", "inference_ms", "p50"), ("summary", "inference_ms", "p50")),
        ("p90 (ms)", ("summary", "inference_ms", "p90"), ("summary", "inference_ms", "p90")),
        ("p95 (ms)", ("summary", "inference_ms", "p95"), ("summary", "inference_ms", "p95")),
        ("FPS equivalente", ("summary", "fps_equivalent_from_inference_mean"), ("summary", "fps_equivalent_from_inference_mean")),
        ("Postproceso medio (ms)", ("summary", "postprocess_ms", "mean"), ("summary", "postprocess_ms", "mean")),
        ("NMS medio (ms)", ("summary", "nms_ms", "mean"), ("summary", "nms_ms", "mean")),
        ("End-to-end medio (ms)", ("summary", "end_to_end_ms", "mean"), ("summary", "end_to_end_ms", "mean")),
        ("PSS final (KiB)", ("memory_after", "total_pss_kb"), ("memory_after", "total_pss_kb")),
        ("Térmico final", ("thermal_after", "status_name"), ("thermal_after", "status_name")),
        ("Eventos por provider", ("provider_profile", "provider_event_counts"), ("provider_profile", "provider_event_counts")),
        ("Error", ("error",), ("error",)),
    ]
    lines = ["| Métrica | CPU/XNNPACK | NNAPI |", "|---|---:|---:|"]
    for label, cpu_keys, nnapi_keys in rows:
        lines.append(f"| {label} | {fmt(nested(cpu, *cpu_keys))} | {fmt(nested(nnapi, *nnapi_keys))} |")
    output = "\n".join(lines) + "\n"
    path = RESULTS / "comparison_latest.md"
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    print(output)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
