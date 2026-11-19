# Etapa 11 – Ajuste fino con todas las fotos de Lima (2026-11-19)

Desde Etapa 9 best.pt, 50 épocas, lr0 0.002. Lima: 83 fotos antiguas (mismo split) + 38 nuevas (CVAT) -> train 86 / val 18 / test 17 (test sin usar).

mAP@50 val Lima (18 fotos, 24 objetos):

| Modelo | Etapa 9 | Etapa 10 ep50 | Etapa 11 best (ep1) | Etapa 11 ep50 |
|---|---|---|---|---|
| LMD-YOLO | 27.9 | 55.2 | 36.5 | 50.4 |
| YOLOv8n | 31.8 | 49.2 | 39.7 | 42.8 |

LMD por clase (Etapa 11 ep50 vs Etapa 10 ep50): D00 48.1/61.4, D10 63.1/54.0, D20 54.5/66.6, D40 35.9/38.9.
LMD P/R val Lima: Etapa 11 ep50 66.9/46.9; Etapa 10 ep50 70.5/46.4.
val RDD LMD: Etapa 9 56.5 -> Etapa 10 52.8 / Etapa 11 52.3. val China Etapa 11 ep50: 74.2.
val_total mAP50 plano (~0.52) desde la época 8. 44.1 s/época (LMD).
Conclusión: más fotos de Lima (38, solo 2 baches) no mejoraron sobre Etapa 10; validación pequeña y ruidosa.

## Modelo final: promedio de pesos (soup) 50/50 de las etapas 10 y 11

Elegido por Rodrigo el 2026-11-19. Detalle en `soup/README.md`; pesos en `soup/weights/lmd_yolo_soup50.pt`
(creados con `colab_stage6/scripts/stage11_soup.py`) y ONNX FP32 640 en `export/` (sha256 eb739e57…, misma detección
principal que PyTorch en 39/39 fotos de validación). Es el modelo de la app ViaDeep-AppMobile.
