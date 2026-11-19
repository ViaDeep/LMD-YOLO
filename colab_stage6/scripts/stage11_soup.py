"""Stage 11 'model soup': average the weights of the Stage-10 and Stage-11 models (epoch 50, last.pt).

Both fine-tunings start from the same Stage-9 best.pt, so their weights can be averaged into one model with the same
size and speed (Wortsman et al., 2022, "Model soups"). alpha is the share of Stage 11: 0 = Stage 10, 1 = Stage 11.
Every soup is evaluated on the same validation sets as Stage 11 (Lima, China, other countries). The TEST splits are
not read. Results go to <results-root>/soup/.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import stage11
from stage6 import sha256

ALPHAS = {"lmd_yolo": (0.25, 0.5, 0.75), "yolov8n": (0.5,)}


def make_soup(path10: Path, path11: Path, alpha: float, out: Path) -> Path:
    a = torch.load(path10, map_location="cpu", weights_only=False)
    b = torch.load(path11, map_location="cpu", weights_only=False)
    sa, sb = a["model"].float().state_dict(), b["model"].float().state_dict()
    if sa.keys() != sb.keys():
        raise AssertionError("Los dos modelos no tienen la misma estructura")
    mixed = {k: ((1 - alpha) * v + alpha * sb[k]) if v.is_floating_point() else v for k, v in sa.items()}
    model = copy.deepcopy(a["model"])
    model.load_state_dict(mixed)
    checkpoint = {k: a[k] for k in ("date", "version", "license", "docs", "train_args", "git") if k in a}
    checkpoint.update(epoch=-1, best_fitness=None, model=model.half(), ema=None, updates=None, optimizer=None,
                      scaler=None, soup={"stage10_last_sha256": sha256(path10), "stage11_last_sha256": sha256(path11),
                                         "alpha_stage11": alpha})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=stage11.CONFIG_PATH)
    args = parser.parse_args()
    stage11.CONFIG_PATH = stage11.stage6.CONFIG_PATH = stage11.stage8.CONFIG_PATH = args.config.expanduser().resolve()
    root = args.results_root.expanduser().resolve()
    cfg = stage11.load_config()
    stage11.restore(root, full_hash=False)
    soup_root = root / "soup"
    summary_path = soup_root / "soup_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for name, alphas in ALPHAS.items():
        spec = cfg["stage10_weights"][name]
        path10 = Path(spec["drive_path"])
        if spec.get("sha256") and sha256(path10) != spec["sha256"]:
            raise AssertionError(f"El last.pt de la etapa 10 de {name} no coincide con el registrado")
        path11 = root / "training" / name / "weights" / "last.pt"
        for alpha in alphas:
            key = f"{name}_alpha{int(alpha * 100):02d}"
            if key in summary:
                print(f"{key}: ya evaluado")
                continue
            weights = make_soup(path10, path11, alpha, soup_root / "weights" / f"{key}.pt")
            report = stage11.evaluate(weights, root, soup_root / "eval" / key, cfg)
            summary[key] = {"weights": str(weights), "sha256": sha256(weights), "alpha_stage11": alpha, **report}
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"{key}: " + ", ".join(f"{g} mAP@50 {r['metrics']['metrics/mAP50(B)']:.3f}" for g, r in report.items()))


if __name__ == "__main__":
    main()
