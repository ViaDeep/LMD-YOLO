# Etapa 5 — overfitting controlado de LMD_YOLO TRAIN

## Diagnóstico

**INCONCLUSIVE.** La reproducción canónica demuestra capacidad real de
aprendizaje: las tres losses descienden con claridad, las métricas crecen de
forma coherente después del warmup, las cuatro clases producen detecciones y
no hay NaN/Inf. Sin embargo, al límite solicitado de 50 épocas el recall de
`best.pt` es 0.4091 y las comparativas visuales todavía muestran omisiones.
Esto prueba aprendizaje, pero no una memorización suficientemente completa de
las 64 imágenes. No se extendió el entrenamiento ni se alteró la arquitectura.

Estas métricas no representan generalización: train y val son exactamente las
mismas 64 imágenes del split train original.

## Modelo auditado

- YAML: `C:\TP1\models\yolov8n_leyolo_nano_mgconv_detect_dbb_stage3.yaml`
- Inicialización: checkpoint TRAIN canónico de etapa 3
  `C:\TP1\runs\detect\leyolo_nano_mgconv_dbb_stage3_smoke\weights\best.pt`
- Backbone: LeYOLO-Nano.
- Neck: dos MGConv `groups=2`, posiciones 21 y 24.
- Head: DetectDBB, exactamente seis DBB TRAIN multirrama.
- Clases: D00, D10, D20, D40.
- Strides: 8, 16, 32.
- Parámetros TRAIN: 3,073,884.
- GFLOPs TRAIN a 640: 8.56617.
- No se llamó `switch_to_deploy()` ni se utilizó el modelo DEPLOY/ONNX.
- El checkpoint final se volvió a cargar desde disco y pasó la misma auditoría
  estructural multirrama.

## Subconjunto

- Seed: `20260923`.
- Selección reproducible: shuffle con seed fija; 24 imágenes únicas que
  contienen D40, después 16 D20, 12 D10 y 12 D00, siempre sin reemplazo.
- Origen exclusivo: `data/dataset/images/train` y labels correspondientes.
- Copias separadas: `C:\TP1\data\overfit64`.
- Validación: 64 imágenes, 64 labels, cero imágenes corruptas, cero fondos,
  coordenadas YOLO finitas y normalizadas, clases exclusivamente 0–3.

| Clase | Objetos | Imágenes que contienen la clase |
|---|---:|---:|
| D00 | 49 | 31 |
| D10 | 25 | 19 |
| D20 | 34 | 25 |
| D40 | 34 | 24 |
| **Total** | **142** | — |

La lista exacta, rutas originales, hashes y conteo por imagen están en
`manifest.csv` y `manifest.json`.

## Configuración de entrenamiento

| Opción | Valor |
|---|---|
| épocas | 50, sin extensión automática |
| `imgsz` | 640 |
| batch | 8 |
| dispositivo | CPU |
| optimizer | AdamW |
| `lr0` / `lrf` | 0.001 / 0.1 |
| beta1 (`momentum`) | 0.9 |
| weight decay | 0.0005 |
| warmup | 3 épocas; momentum 0.8; bias LR 0.01 |
| seed | 20260923; deterministic=True |
| patience | 100 |
| augmentations | mosaic, mixup, cutmix, copy-paste, HSV, flip, translate, scale, degrees, shear, perspective y BGR desactivados |
| train / val | las mismas 64 imágenes |

AdamW con LR 1e-3 es una elección conservadora y explícita para optimizar un
detector ya inicializado desde el checkpoint TRAIN de etapa 3 sobre un lote
pequeño. No se usó `optimizer=auto` ni se reajustaron hiperparámetros durante
la ejecución para favorecer el resultado.

## Evolución

| Época | box loss | cls loss | DFL loss | P | R | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.22135 | 5.84039 | 4.30346 | 0.00042 | 0.05147 | 0.00012 | 0.00004 |
| 5 | 3.83371 | 5.10247 | 3.93343 | 0.00039 | 0.05147 | 0.00010 | 0.00003 |
| 10 | 3.61285 | 4.86106 | 3.61326 | 0.00038 | 0.05147 | 0.00009 | 0.00003 |
| 20 | 2.44655 | 3.44002 | 2.95342 | 0.00057 | 0.08373 | 0.00030 | 0.00014 |
| 25 | 2.08164 | 2.95202 | 2.53677 | 0.76611 | 0.04952 | 0.11439 | 0.06212 |
| 30 | 1.75463 | 2.54950 | 2.25219 | 0.76001 | 0.28663 | 0.30769 | 0.20366 |
| 35 | 1.59963 | 2.31463 | 2.08104 | 0.67741 | 0.41211 | 0.43532 | 0.28346 |
| 40 | 1.35074 | 1.97869 | 1.85179 | 0.81202 | 0.41368 | 0.48435 | 0.34232 |
| 45 | 1.30788 | 1.89189 | 1.81503 | 0.89500 | 0.40823 | 0.49972 | 0.36340 |
| 50 | 1.18378 | 1.79883 | 1.71242 | 0.92443 | 0.40901 | 0.51420 | 0.38252 |

Entre épocas 1 y 50, box loss cayó 72.0%, cls loss 69.2% y DFL loss
60.2%. El CSV conserva las 50 filas completas.

## Validación explícita de best.pt

`best.pt` se cargó desde disco y se evaluó nuevamente sobre las mismas 64
imágenes:

- Precision: **0.933498**
- Recall: **0.409056**
- mAP50: **0.516547**
- mAP50-95: **0.383061**

| Clase | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| D00 | 0.86585 | 0.38776 | 0.48066 | 0.32630 |
| D10 | 0.88181 | 0.40000 | 0.46503 | 0.26368 |
| D20 | 0.98633 | 0.52941 | 0.69990 | 0.60292 |
| D40 | 1.00000 | 0.31906 | 0.42060 | 0.33935 |

D20 es la clase más aprendida. D40 no está bloqueada (precision 1.0,
mAP50 0.4206), pero su recall 0.3191 es el menor y confirma que aún hay
objetos no memorizados.

## Tiempo, warnings y artefactos

- Entrenamiento y validación interna: 2,443.52 s (40 min 43.5 s).
- Validación explícita de `best.pt`: 12.10 s.
- Total medido por el script: 2,457.28 s (40 min 57.3 s).
- Sin errores, NaN o Inf.
- Warning no funcional: acceso inicial a imágenes reportado como lento; las
  lecturas posteriores desde cache fueron rápidas.
- Aviso informativo: Ultralytics 8.4.160 estaba disponible; se mantuvo la
  versión local 8.4.153 para reproducibilidad.
- La validación de Ultralytics usa una copia con fusión Conv+BN en memoria para
  inferencia. No se guardó esa copia ni se reparametrizaron los DBB; el
  `best.pt` en disco conserva los seis DBB TRAIN multirrama.

Artefactos principales:

- Subset/YAML: `C:\TP1\data\overfit64`
- Manifest: `C:\TP1\outputs\stage5_overfit64\manifest.csv`
- Preflight estructural/dataset: `C:\TP1\outputs\stage5_overfit64\preflight.json`
- Mosaico de labels: `C:\TP1\outputs\stage5_overfit64\labels_mosaic.jpg`
- Resumen máquina-legible: `C:\TP1\outputs\stage5_overfit64\training_summary.json`
- Curvas y CSV: `C:\TP1\runs\detect\lmd_stage5_overfit64`
- `best.pt`: `C:\TP1\runs\detect\lmd_stage5_overfit64\weights\best.pt`
- `last.pt`: `C:\TP1\runs\detect\lmd_stage5_overfit64\weights\last.pt`
- Validación explícita: `C:\TP1\runs\detect\lmd_stage5_overfit64_best_val`
- Comparativas individuales: `C:\TP1\outputs\stage5_overfit64\predictions`
- Mosaico GT/predicción: `C:\TP1\outputs\stage5_overfit64\predictions_mosaic.jpg`

