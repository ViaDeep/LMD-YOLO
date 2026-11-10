# Resultados de la etapa 10 (fotos de Lima)

Ajuste fino de los dos modelos desde su `best.pt` de la etapa 9, con fotos de calles de Lima tomadas con celular
desde la vereda (`colab_stage6/configs/stage10_lima.yaml`, `scripts/stage10.py`, `stage10_colab.ipynb`).
50 épocas, SGD, lr0 0,002, imgsz 640.

- Entrenamiento: 59 fotos de Lima (repetidas 6 veces) + 1200 fotos fijas del entrenamiento de la etapa 8, para no
  olvidar RDD2022.
- Validación: 12 fotos de Lima (16 daños) + China + otros países. División fija de las 83 fotos de Lima en
  `colab_stage6/configs/lima_split.json` (59/12/12). El test de Lima (12 fotos) y los demás test no se usaron.
- La selección de `best.pt` usa la validación total (casi toda RDD2022), por eso `best.pt` es la época 1 en ambos
  modelos. El modelo elegido para la app es el de la **época 50** (`last.pt`).

| mAP@50 (val) | LMD E9 | LMD E10 ép. 1 | LMD E10 ép. 50 | v8n E9 | v8n E10 ép. 1 | v8n E10 ép. 50 |
|---|---|---|---|---|---|---|
| Lima | 37,4 | 50,3 | 62,4 | 36,1 | 49,7 | 58,9 |
| China | 79,7 | 77,3 | 74,3 | 78,8 | 75,8 | 73,7 |
| Otros países | 47,2 | 45,8 | 44,3 | 48,1 | 46,4 | 44,3 |
| Total | 56,2 | 54,0 | 52,8 | 57,3 | 55,1 | 53,2 |

Lectura: con más épocas mejora Lima y baja RDD2022 (sobreajuste a 59 fotos). La mejora en Lima se mide con solo
16 daños, así que es una señal preliminar.

## Contenido

- `training/lmd_yolo/`: `results.csv`, `stage10_summary.json`, `lmd_yolo_last.json` (métricas de la época 50 por
  grupo y por clase) y `weights/last.pt` (época 50, sha256 `ad782f56…`).
- `training/yolov8n/`: `yolov8n_last.json`.
- `baseline_stage9.json`: modelos de la etapa 9 sobre la misma validación.
- `comparison/`: `comparison.json` y `learning_curves.png`.
- `export/`: `lmd_yolo_stage10_ep50_deploy_fp32.onnx` (DBB reparametrizado, FP32, 640, sin NMS; entrada `images`
  [1,3,640,640], salida `output0` [1,8,8400]), `export_report.json` (diferencia máxima con PyTorch 0,0028; detección
  principal igual en 39/39 fotos de validación de China) y `export_stage10.py`, que lo genera. Es el mismo archivo
  que usa la app ViaDeep-AppMobile en `assets/models/lmd-yolo.onnx`.
