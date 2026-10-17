# Stage 6 — YOLOv8n vs LMD_YOLO en Colab T4

Este paquete es autocontenido y separado del proyecto canónico. No contiene
runs antiguos, ONNX, artefactos DEPLOY, mobile/NNAPI, `.venv` ni overfit64.

## Decisión de inicialización

Ambos modelos parten **desde cero** y reciben la misma seed (`42`). Usar
`yolov8n.pt` daría pretraining COCO solo al baseline; usar el checkpoint LMD
de smoke introduciría una época de exposición a RDD2022 solo para LMD. No
existe un checkpoint externo equivalente para LeYOLO/MGConv/DetectDBB.

Los dos checkpoints se incluyen bajo `reference_weights/` únicamente para
auditoría/procedencia. `scripts/stage6.py` construye siempre desde YAML y no
acepta un argumento de pesos.

## Configuración común

- 20 épocas, 640 px, batch 16, seed 42, AMP.
- SGD: LR 0.01, momentum 0.937, weight decay 0.0005, `lrf=0.01`.
- Warmup de 3 épocas; patience 100.
- Augmentation normal idéntica: HSV 0.015/0.7/0.4, translate 0.1,
  scale 0.5, horizontal flip 0.5 y mosaic 1.0 (cerrado en las últimas
  10 épocas). Mixup/cutmix/copy-paste permanecen en 0 para ambos.
- Train: 3,082; val: 385; test: 386.
- TEST está presente para integridad, pero el script no ofrece ninguna acción
  que lo evalúe.

La configuración completa está en `configs/comparison.yaml`.

## Qué subir y dónde

Subir **un único archivo**:

`colab_stage6.zip`

a esta ruta exacta de Google Drive:

`MyDrive/TP1_LMD/stage6/packages/colab_stage6.zip`

Crear si no existen las carpetas `TP1_LMD/stage6/packages`. Los resultados se
guardarán automáticamente en:

`MyDrive/TP1_LMD/stage6/results/`

El notebook extrae una copia de trabajo en `/content/colab_stage6` para no
entrenar leyendo miles de imágenes directamente desde Drive. Checkpoints,
`results.csv`, curvas, preflight y smoke se escriben en Drive.

## Ejecución

1. Abrir `stage6_colab.ipynb` en el notebook Colab ya conectado a una T4.
2. Ejecutar, en orden, solo la sección **PREFLIGHT + SMOKE**.
3. Confirmar que `preflight.json` y `smoke_gate.json` indican `PASS` para
   ambos modelos y revisar la VRAM registrada.
4. Detenerse. No ejecutar las celdas de 20 épocas hasta revisar el smoke.
5. Para continuar posteriormente, cambiar manualmente
   `RUN_LONG_TRAINING = True` en la celda-gate.
6. Ejecutar primero YOLOv8n y luego LMD_YOLO. Cada corrida rechaza sobrescribir
   resultados existentes.
7. Ejecutar la celda de comparación solo cuando ambas hayan terminado.

Reiniciar Colab no invalida el flujo: volver a montar Drive, extraer el ZIP y
usar el mismo directorio persistente de resultados. No se debe repetir una
corrida ya existente; conservarla o moverla explícitamente antes de reintentar.

## Integridad

- `manifests/dataset_manifest.json`: SHA-256 individual de todas las imágenes
  y labels, conteos y digest combinado.
- `manifests/package_manifest.json`: SHA-256 de todos los archivos del paquete.
- `scripts/manifests.py verify-package`: verificación completa tras extraer.
- El preflight vuelve a comprobar todo el dataset, las clases, splits,
  arquitectura, forward/backward CUDA y procedencia de Ultralytics.

No instalar `ultralytics` mediante pip. La copia autorizada está en
`vendor/ultralytics` y el preflight rechaza cualquier otra procedencia.

## Etapa 7 — 100 épocas

Misma configuración que la etapa 6 salvo `epochs: 100`, en `configs/stage7_100_epochs.yaml`. Se usa
el notebook `stage7_colab.ipynb`, que pasa `--config` a cada acción. Ambos modelos parten de cero con
seed 42, igual que en la etapa 6; los resultados de la etapa 6 no se modifican.

- ZIP: `MyDrive/TP1_LMD/stage7/packages/colab_stage6.zip`.
- Resultados: `MyDrive/TP1_LMD/stage7/results/`.
- Si Colab se desconecta, `stage6.py train --model <modelo> --resume` retoma la corrida desde su
  `last.pt` con el mismo calendario de LR. El tiempo del resumen sale de la columna `time` de
  `results.csv`, que Ultralytics acumula entre sesiones.
- El resumen exige exactamente las épocas del config y registra el config usado y si hubo reanudación.

El paquete del repositorio guarda los labels con fin de línea LF, pero los manifiestos se generaron con
CRLF. Antes de comprimir hay que restaurar CRLF en los archivos cuyo hash coincide así, o el preflight
rechazará el dataset.


## Etapa 8 — Más países de RDD2022

Ajuste de 50 épocas de los dos modelos desde su `best.pt` de la etapa 7, con `configs/stage8_paises.yaml` y el
notebook `stage8_colab.ipynb` (script `scripts/stage8.py`). El objetivo es reducir la diferencia de dominio: la
etapa 7 solo vio China (la mitad desde dron) y falla con fotos de calle tomadas con celular.

- Datos: train y val de China de la etapa 7, más hasta 2000 imágenes etiquetadas por país de Japón, India,
  República Checa y Estados Unidos (cámaras montadas en auto). Solo se usan imágenes con al menos un
  D00/D10/D20/D40, la misma regla de `scripts/convert_voc_to_yolo.py`. Cada país se divide 80/10/10 con seed 42.
- El TEST de China no se copia al dataset de la etapa 8. El 10 % de test de los países nuevos queda en
  `test_paises_bloqueado`, fuera del YAML de datos; ninguna acción lo evalúa.
- Descarga: primero `MyDrive/TP1_LMD/rdd2022/RDD2022_<País>.zip` si ya existe, luego el enlace oficial de
  CRDDC2022 en S3 y, si responde 403, FigShare (artículo 21431547). Si nada funciona, subir a mano el ZIP del país
  a esa carpeta de Drive.
- `prepare` guarda en `MyDrive/TP1_LMD/stage8/results/` el dataset armado (`dataset_stage8.zip`) y su manifiesto
  con SHA-256 por archivo; las sesiones siguientes lo restauran y verifican con `restore`.
- `baseline` evalúa los modelos de la etapa 7 sobre la validación nueva, por separado en China y en los otros
  países. `compare` pone al lado los valores de la etapa 8.
- `train --resume` retoma una corrida cortada desde su `last.pt`.
