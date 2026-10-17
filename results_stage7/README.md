# Resultados de la etapa 7 (100 épocas)

Entrenamiento desde cero de LMD-YOLO y YOLOv8n con la misma configuración que la etapa 6, pero 100 épocas
(`colab_stage6/configs/stage7_100_epochs.yaml`). Métricas sobre el conjunto de **validación**; el conjunto de
prueba no se ha usado y se evaluará una sola vez con el modelo final.

| Métrica (val) | LMD-YOLO | YOLOv8n |
|---|---|---|
| mAP@50 | 0,745 (mejor época 99) | 0,690 (época 100) |
| mAP@50:95 | 0,441 | 0,409 |
| Precisión | 0,722 | 0,731 |
| Recall | 0,684 | 0,634 |
| Parámetros | 3,07 M | 3,01 M |
| GFLOPs | 8,57 | 8,19 |

mAP@50 por clase (LMD-YOLO / YOLOv8n): D00 0,764 / 0,757 · D10 0,818 / 0,769 · D20 0,734 / 0,701 · D40 0,664 / 0,534.

## Contenido

- `training/lmd_yolo/`: `results.csv`, `stage6_summary.json` y `weights/best.pt` (sin `last.pt` ni checkpoints intermedios).
- `training/yolov8n/`: `results.csv` y `stage6_summary.json`.
- `comparison/`: `comparison.json` y `learning_curves.png`.
- `export/`: `lmd_yolo_stage7_deploy_fp32.onnx` (DBB reparametrizado, FP32, 640, sin NMS; entrada `images`
  [1,3,640,640], salida `output0` [1,8,8400]) y `export_report.json` (diferencia máxima con PyTorch 0,0027).
  Es el mismo archivo que usa la app ViaDeep-AppMobile en `assets/models/lmd-yolo.onnx`.

La latencia de 113,77 ms en Galaxy S23 se midió con el checkpoint de 1 época y está pendiente de repetir con este modelo.
