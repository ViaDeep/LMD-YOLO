"""Export Stage 10 LMD-YOLO (epoch 50, last.pt) to ONNX FP32 640 with DBB in deploy mode, and check parity.

Run from the repository root: python results_stage10/export/export_stage10.py
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'colab_stage6'; sys.path.insert(0, str(ROOT / 'vendor'))
from ultralytics import YOLO
from ultralytics.nn.modules import DiverseBranchBlock
from ultralytics.data.augment import LetterBox
import cv2, onnx, onnxruntime as ort

SRC = REPO / 'results_stage10/training/lmd_yolo/weights/last.pt'
OUT = REPO / 'results_stage10/export'; OUT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
src_sha = sha(SRC); assert src_sha.startswith('ad782f56'), src_sha
ckpt = torch.load(SRC, map_location='cpu', weights_only=False)
print('epoch in ckpt:', ckpt.get('epoch'), 'train_args epochs:', ckpt.get('train_args', {}).get('epochs'))
local = OUT / 'lmd_yolo_stage10_ep50.pt'; shutil.copy(SRC, local)  # export writes next to the weights

model = YOLO(str(local), task='detect', verbose=False)
net = model.model.float().eval()
imgs = sorted((ROOT / 'dataset/images/val').glob('*.jpg'))[:39]
lb = LetterBox((640, 640), auto=False, stride=32)
def tensor(p):
    im = lb(image=cv2.imread(str(p)))[..., ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(im)).float().div(255).unsqueeze(0)
X = [tensor(p) for p in imgs]
with torch.no_grad():
    ref = [net(x)[0] if isinstance(net(x), (list, tuple)) else net(x) for x in X]
dbbs = [m for m in net.modules() if isinstance(m, DiverseBranchBlock)]
assert len(dbbs) == 6
for m in dbbs: m.switch_to_deploy()
with torch.no_grad():
    dep = [(lambda o: o[0] if isinstance(o, (list, tuple)) else o)(net(x)) for x in X]
deploy_max = max(float((a - b).abs().max()) for a, b in zip(ref, dep))
print('deploy vs train max abs', deploy_max)

f = model.export(format='onnx', imgsz=640, batch=1, simplify=False, dynamic=False, nms=False)
onnx_path = OUT / 'lmd_yolo_stage10_ep50_deploy_fp32.onnx'; shutil.move(f, onnx_path)
onnx.checker.check_model(str(onnx_path))
sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
diffs, matches, rows = [], 0, []
for p, x, r in zip(imgs, X, ref):
    o = sess.run(None, {'images': x.numpy()})[0]
    d = np.abs(o - r.numpy()); diffs.append((d.max(), d.mean()))
    def top(a):
        s = a[0, 4:, :]; i = np.unravel_index(s.argmax(), s.shape); return int(i[0]), float(s[i]), int(i[1])
    to, tr = top(o), top(r.numpy())
    ok = to[0] == tr[0] and to[2] == tr[2] and abs(to[1] - tr[1]) < 1e-3
    matches += ok; rows.append({'image': p.name, 'onnx': to, 'torch': tr, 'match': ok})
report = {
    'source_checkpoint': 'results_stage10/training/lmd_yolo/weights/last.pt (epoch 50)', 'source_sha256': src_sha,
    'ultralytics': '8.4.153 (vendored)', 'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
    'deploy_max_abs_vs_train': deploy_max,
    'onnx_max_abs_vs_train': float(max(d[0] for d in diffs)), 'onnx_mean_abs_vs_train': float(np.mean([d[1] for d in diffs])),
    'inputs': [[i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]] for i in onnx.load(str(onnx_path)).graph.input],
    'outputs': [[o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]] for o in onnx.load(str(onnx_path)).graph.output],
    'onnx_file': onnx_path.name, 'onnx_bytes': onnx_path.stat().st_size, 'onnx_sha256': sha(onnx_path),
    'parity_images': len(imgs), 'parity_top_detection_matches': matches, 'parity_rows': rows,
}
(OUT / 'export_report.json').write_text(json.dumps(report, indent=2))
local.unlink()
print(json.dumps({k: v for k, v in report.items() if k != 'parity_rows'}, indent=2))
