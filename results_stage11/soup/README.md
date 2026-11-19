# Promedio de pesos Etapa 10 + Etapa 11 (model soup), 2026-11-19

Peso = (1-a)·Etapa10 last.pt + a·Etapa11 last.pt (ambos ajustes parten del best.pt de la Etapa 9). Sin entrenamiento, TEST sin usar. Script: LMD-YOLO colab_stage6/scripts/stage11_soup.py (commit e9cbaf4).

LMD-YOLO, mAP@50 (P / R) en %:

| Validación | Etapa 10 ep50 | Etapa 11 ep50 | a=0.25 | a=0.50 | a=0.75 |
|---|---|---|---|---|---|
| Lima (18 fotos) | 55.2 (70.5/46.4) | 50.4 (66.9/46.9) | 55.3 | **56.8 (58.2/64.0)** | 55.4 |
| RDD (China+países) | 52.8 | 52.3 | 53.3 | 52.8 | 52.7 |
| China | 74.3 | 74.2 | 74.8 | 74.7 | 74.3 |

Lima por clase a=0.50 (Etapa 10 / Etapa 11): D00 61.7 (61.4/48.1), D10 63.1 (54.0/63.1), D20 64.5 (66.6/54.5), D40 37.9 (38.9/35.9).
YOLOv8n a=0.50: Lima 50.1 (Etapa 10 49.2, Etapa 11 42.8), RDD 53.7.
