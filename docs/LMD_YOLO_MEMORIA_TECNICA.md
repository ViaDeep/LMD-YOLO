# Memoria técnica maestra del proyecto LMD_YOLO

**Proyecto:** detección de daños viales D00, D10, D20 y D40 y evaluación de inferencia on-device  
**Raíz auditada:** `C:\TP1`  
**Fecha de corte de la auditoría:** 2026-09-23  
**Naturaleza del documento:** reconstrucción cronológica basada exclusivamente en artefactos locales; no se ejecutaron entrenamientos, benchmarks, conversiones ni inferencias durante esta auditoría.

> **Regla de interpretación.** “Validado” significa respaldado por un artefacto local identificable. No significa réplica exacta del paper, precisión definitiva, superioridad general ni portabilidad a otros teléfonos. Las fechas de la cronología proceden de las marcas de modificación de los archivos porque `C:\TP1` no es un repositorio Git.

## 1. Propósito del documento

Esta memoria reconstruye el trabajo realizado para determinar si una reproducción funcional de LMD_YOLO puede entrenarse para cuatro clases de daño vial y ejecutarse localmente en un smartphone. Separa explícitamente:

- lo atribuido al paper por la documentación del proyecto;
- lo implementado en `C:\TP1`;
- las decisiones propias necesarias por ambigüedad del paper;
- los resultados experimentales realmente presentes;
- lo pendiente o no demostrable con el material local.

El documento es una fuente maestra de trazabilidad, no un capítulo final de tesis. Las métricas de smoke, Overfit64 y el checkpoint on-device no deben confundirse con una evaluación final de generalización.

## 2. Alcance, método y nivel de evidencia

Se inspeccionaron `docs`, `scripts`, `tests`, `models`, `outputs`, `runs`, `mobile_benchmark`, `colab_stage6`, `results_stage6`, `data`, `paper`, `third_party/ultralytics` en sus módulos personalizados, archivos raíz, el contenido listado de `colab_stage6.zip` y el PDF externo `C:\Users\Usuario\Downloads\full_text (11).pdf`. Se excluyeron `.venv`, caches y la mayor parte del código vendorizado no relacionado. El inventario inicial contenía 33.776 archivos visibles mediante `rg --files`; a él se suman los 129 artefactos recuperados de Stage 6. El PDF se utilizó como fuente bibliográfica, no como instrucciones operativas.

La evidencia se clasifica así:

| Nivel | Significado |
|---|---|
| E1 | Resultado máquina-legible: JSON, CSV, checkpoint, ONNX o manifest con hash. |
| E2 | Configuración o código que define lo que debía ejecutarse. |
| E3 | Informe Markdown elaborado durante la etapa y consistente con E1/E2. |
| E4 | Inferencia razonable, indicada como tal; no se presenta como medición. |

No se encontró un `AGENTS.md`, historial Git ni logs completos de consola. La cronología usa artefactos y `LastWriteTime`, por lo que refleja el orden material disponible, no necesariamente cada interacción humana.

## 3. Contexto y objetivo técnico

El problema es detección de objetos con cuatro clases RDD2022:

| ID | Clase |
|---:|---|
| 0 | D00 |
| 1 | D10 |
| 2 | D20 |
| 3 | D40 |

La estrategia observada fue incremental: preparar datos; crear un baseline YOLOv8n; reemplazar el backbone por LeYOLO-Nano; introducir MGConv; introducir Detect_DBB; comprobar entrenamiento; reparametrizar a DEPLOY; exportar ONNX; fijar un preprocesamiento PC–Android; medir CPU/XNNPACK y probar NNAPI; comprobar capacidad de aprendizaje con Overfit64; y preparar una comparación controlada Stage 6 desde cero.

**Fuentes:** `data/data.yaml`; `docs/architecture_decisions.md`; `docs/stage3_detect_dbb.md`; `colab_stage6/README.md`.

## 4. Paper y arquitectura de referencia

### 4.1 Fuente primaria verificada

Se revisó el PDF externo `C:\Users\Usuario\Downloads\full_text (11).pdf`: **“LMD-YOLO: A Lightweight Road Damage Detection Algorithm Based on YOLOv8”**, Shuai He, Ye Yuan y Bingyang Yin, DOI `10.1109/ACCESS.2025.3557938`. La carpeta local `paper/` continúa vacía; por trazabilidad futura conviene conservar una copia estable y su SHA-256, pero la ausencia de esa copia ya no impide contrastar el contenido. Las referencias de página siguientes usan la paginación impresa del artículo.

### 4.2 A) Lo que afirma y lo que no especifica el paper

- El modelo completo combina backbone LeYOLO, `mg_conv` en el neck y `Detect_DBB` en el head, con salidas P3/P4/P5 (pp. 4–7, secciones III.C–III.F, Figs. 5–9).
- LeYOLO se presenta como backbone ligero basado en inverted bottlenecks y conexiones residuales dinámicas; el artículo lo motiva para despliegue móvil/embebido (pp. 4–5). No publica variante, YAML, tabla completa de canales, código ni pesos que permitan reconstrucción bit a bit.
- `mn_conv` se describe como Conv2d–BatchNorm–SiLU y admite convolución agrupada cuando `groups>1` (p. 5). No se da el valor efectivo de grupos usado por LMD_YOLO.
- `mg_conv` reemplaza convoluciones del neck, cambia SiLU por GELU y añade una conexión residual (pp. 5–6, Fig. 7). No especifica grupos, índices exactos, kernel, stride, canales, bias ni detalles completos del shortcut.
- DBB usa ramas multiescala durante entrenamiento y se fusiona en una sola convolución para inferencia (p. 6, sección III.E.1). La ecuación de la rama promedio presenta `AvgPool(I)` seguido de una transformación 1×1; el artículo no ofrece código suficiente para resolver todos los detalles de la implementación.
- `Detect_DBB` sustituye el head de YOLOv8 (pp. 6–7, sección III.E.2, Fig. 8), pero no especifica seis DBB ni sus posiciones internas exactas.
- El entorno publicado fue Ubuntu 20.04, Python 3.9, PyTorch, CPU Intel Xeon Gold 5218 y GPU NVIDIA A100 40 GB (p. 8, Tabla 1). El entrenamiento publicado usó 200 épocas, batch 64, imagen 640, SGD, LR inicial 0,01, momentum 0,937 y annealing cosenoidal (p. 8, Tabla 2 y texto asociado).
- La Tabla 3 (p. 9) informa YOLOv8n: 3,01 M, 8,2 G, 69 FPS; YOLOv8n-LeYOLO: 1,39 M, 3,9 G, 61 FPS. La Tabla 4 (p. 10) informa `mg_conv`: 1,34 M, 3,8 G, 60 FPS, P 0,849, R 0,780, mAP50 0,850 y mAP50-95 0,505.
- La Tabla 5 (p. 10) informa para LeYOLO+MGConv+DBB: 1,70 M, 3,8 G, 63 FPS, P 0,823, R 0,830, mAP50 0,893 y mAP50-95 0,556. La Tabla 6 (p. 11) repite esos resultados para LMD_YOLO frente a YOLOv8n: 0,862/0,768/0,856/0,524, 8,2 G, 69 FPS, 3,01 M.
- El paper afirma orientación y aptitud para dispositivos móviles, embebidos, portátiles y edge (resumen, introducción y conclusión), pero no presenta benchmark en smartphone, teléfono físico, runtime móvil ni consumo energético. Los FPS se publican bajo el entorno A100/Xeon; no son evidencia de ejecución en teléfono.

**Fuente primaria:** `full_text (11).pdf`, DOI `10.1109/ACCESS.2025.3557938`, pp. 1, 4–11 y 14, Figs. 5–9, Tablas 1–6.

### 4.3 B) Lo implementado en este proyecto

- Port local de `LeYOLOConv` y `LeYOLOInvertedBottleneck` basado en el repositorio público LeYOLO.
- Backbone LeYOLO-Nano materializado en YAML y neck/head YOLOv8n efectivo.
- Dos `MGConv(groups=2)` en las capas YAML 21 y 24.
- `DetectDBB` que reemplaza el primer Conv 3×3 de las torres `cv2` y `cv3` en P3/P4/P5: seis DBB.
- Forma TRAIN multirrama y conversión DEPLOY mediante `switch_to_deploy()`.
- Ultralytics 8.4.153 vendorizado con parser y registro de módulos personalizados.

**Fuentes:** `models/yolov8n_leyolo_nano_stage1.yaml`; `models/yolov8n_leyolo_nano_mgconv_stage2.yaml`; `models/yolov8n_leyolo_nano_mgconv_detect_dbb_stage3.yaml`; `third_party/ultralytics/nn/modules/leyolo.py`; `mg_conv.py`; `detect_dbb.py`; `diverse_branch_block.py`; `colab_stage6/AUDIT.md`.

### 4.4 C) Decisiones propias por ambigüedad

| Decisión propia | Justificación local | No debe afirmarse como hecho del paper |
|---|---|---|
| LeYOLO-Nano frente a Small | Nano fue el candidato local más cercano a las cifras de referencia. | El paper no publica variante, tabla completa, YAML, código ni pesos exactos. |
| Neck YOLOv8n fijo en Stage 1 | Aislar el cambio de backbone. | No demuestra que sea el neck de los autores. |
| MGConv en capas 21 y 24 | Interpretación de la figura sobre el neck descendente. | Los índices dependen del YAML local. |
| Rama principal MGConv 3×3, stride 2, `groups=2` | Herencia de convoluciones reemplazadas y variante no trivial mínima ensayada. | El paper no publica kernel, stride, canales ni número de grupos efectivo. |
| Atajo MGConv Conv 1×1, stride 2, grupos 1, con bias | Necesario para igualar forma; coherencia con el port LeYOLO. | No está especificado por los autores. |
| Seis DBB | Un DBB en el primer Conv de `cv2` y otro en el primero de `cv3` por cada escala. | El paper no especifica seis ni posiciones internas exactas. |
| DBB oficial adaptado | Se preserva la fusión del repositorio DiverseBranchBlock y sus cuatro ramas: K×K, 1×1, 1×1→K×K y 1×1→AvgPool. | El paper no ofrece código ni una descripción inequívoca de esas cuatro ramas; su ecuación de promedio sugiere AvgPool→1×1. |

### 4.5 D) Resultados propios y límite de fidelidad

La arquitectura local es una **reproducción funcional documentada**, no una réplica exacta. Sus valores absolutos no igualan las ablaciones del paper y no se forzaron mediante ajuste retrospectivo. Stage 6 mide 3,073884 M y 8,566170 GFLOPs para la forma TRAIN local, frente a 1,70 M y 3,8 G publicados para LMD_YOLO; la diferencia confirma que la implementación propia no debe presentarse como réplica exacta.

## 5. Cronología verificada

| Fecha local | Etapa | Resultado verificable | Fuente principal |
|---|---|---|---|
| 2026-09-16 | Preparación de dataset | 3.853 imágenes válidas; split 3.082/385/386. | `scripts/*.py`; `data/dataset`; `data/data.yaml` |
| 2026-09-16 | Baseline YOLOv8n | Una época CPU con pesos COCO; validación en `val`. | `runs/detect/train/args.yaml`, `results.csv` |
| 2026-09-18 | Stage 1 LeYOLO | Nano seleccionado; estructura y smoke de una época pasan. | `outputs/stage1_leyolo/*` |
| 2026-09-20 | Stage 2 MGConv | Variante B, grupos 2, se vuelve canónica; smoke pasa. | `outputs/stage2_mgconv*/*` |
| 2026-09-20 | Stage 3 Detect_DBB | Variante de 6 DBB elegida; TRAIN y DEPLOY equivalentes dentro de tolerancia. | `outputs/stage3_detect_dbb/*` |
| 2026-09-21 | Stage 4 ONNX | Exportación FP32 estática validada con ORT CPU. | `outputs/stage4_mobile_export_audit/*` |
| 2026-09-22 | Android | Preprocesamiento canónico, NNAPI abortado y baseline CPU/XNNPACK definitivo. | `mobile_benchmark/results/*`; `docs/*mobile*`, `docs/nnapi*` |
| 2026-09-23 | Stage 5 Overfit64 | Aprendizaje demostrado; memorización completa no alcanzada en 50 épocas. | `outputs/stage5_overfit64/*`; `runs/detect/lmd_stage5_overfit64/*` |
| 2026-09-23 | Stage 6 | Preflight y smoke PASS; entrenamientos controlados de 20 épocas completos; LMD mejora mAP50 y mAP50-95 de validación, con mayor coste. | `results_stage6/results-20260923T224020Z-1-001/results/*` |

## 6. Dataset

### 6.1 Origen observado

El material fuente local contiene:

| Fuente | Imágenes/XML |
|---|---:|
| `data/raw/China_Drone/train` | 2.401 |
| `data/raw/China_MotorBike/train` | 1.977 |
| **Total de XML de entrenamiento inspeccionados** | **4.378** |
| `data/raw/China_MotorBike/test/images` | 500 imágenes, sin uso en los scripts de conversión observados |

`convert_voc_to_yolo.py` toma únicamente los dos directorios `train`, filtra a D00/D10/D20/D40, copia una imagen solo si conserva al menos un objeto y escribe etiquetas YOLO normalizadas.

### 6.2 Filtrado y distribución

El recuento directo de los XML locales arroja:

| Clase/etiqueta | Objetos |
|---|---:|
| D00 | 4.104 |
| D10 | 2.359 |
| D20 | 934 |
| D40 | 321 |
| Repair, ignorada | 1.046 |
| Block crack, ignorada | 3 |
| **Objetos válidos** | **7.718** |
| **Objetos ignorados** | **1.049** |
| **Objetos originales** | **8.767** |

Hay 3.853 imágenes convertidas y 525 XML sin ninguna de las cuatro clases válidas (`4.378 - 3.853`). El par `data/images`/`data/labels` contiene 3.853 archivos por lado.

**Fuentes:** `scripts/check_classes.py`; `scripts/convert_voc_to_yolo.py`; XML en `data/raw/.../annotations/xmls`; `data/images`; `data/labels`.

### 6.3 Split reproducible

`split_dataset.py` ordena los JPG, usa `random.seed(42)`, baraja y corta 80/10/10 mediante enteros. El resultado presente es:

| Split | Imágenes | Labels | Objetos | D00 | D10 | D20 | D40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3.082 | 3.082 | 6.191 | 3.309 | 1.867 | 732 | 283 |
| val | 385 | 385 | 759 | 390 | 255 | 97 | 17 |
| test | 386 | 386 | 768 | 405 | 237 | 105 | 21 |
| **Total** | **3.853** | **3.853** | **7.718** | **4.104** | **2.359** | **934** | **321** |

El manifest de Stage 6 confirma correspondencia uno-a-uno, clases 0–3, coordenadas normalizadas y digest combinado `8ae529f168c13e46d014c4799810becaecdb271128ccd0e40780ff5c713f07eb`.

**Fuentes:** `scripts/split_dataset.py`; `data/dataset`; `colab_stage6/manifests/dataset_manifest.json`; `colab_stage6/AUDIT.md`.

### 6.4 Estado del conjunto TEST

No se encontró evidencia de evaluación del split original `data/dataset/images/test`:

- los cinco `args.yaml` locales de entrenamiento declaran `split: val`;
- los scripts Stage 1–3 validan en `val`;
- Overfit64 usa 64 imágenes procedentes exclusivamente de `train` y configura su propio `test` como alias de esas mismas imágenes, lo que **no** toca el TEST original;
- Stage 6 configura `validation.split: val`, `test_is_forbidden: true` y no ofrece acción de evaluación TEST;
- el script Stage 6 solo recorre TEST durante verificación de integridad/hashes;
- no existe un directorio de resultados de evaluación TEST.

Conclusión precisa: **el TEST original permanece sin evaluación según toda la evidencia local inspeccionada**. No puede demostrarse una ausencia absoluta de uso fuera de `C:\TP1` o en sesiones no recuperadas.

**Fuentes:** `runs/detect/*/args.yaml`; `scripts/train_*`; `scripts/run_stage5_overfit64.py`; `colab_stage6/configs/comparison.yaml`; `colab_stage6/scripts/stage6.py`.

## 7. Baseline inicial YOLOv8n

El baseline inicial no es la comparación controlada Stage 6. Usó `yolov8n.pt`, `pretrained: true`, `cls_remap: true`, CPU, una época, batch 4, imagen 640, workers 8, seed 0, determinismo y validación en `val`. `optimizer: auto` figura en la configuración.

| Métrica, época 1 | Valor |
|---|---:|
| Tiempo reportado | 1.268,71 s |
| train box / cls / DFL | 2,41008 / 3,53879 / 2,12346 |
| val box / cls / DFL | 2,37882 / 3,52817 / 2,14307 |
| Precision | 0,36882 |
| Recall | 0,14916 |
| mAP50 | 0,07786 |
| mAP50-95 | 0,02621 |

Este run prueba un flujo baseline corto con transferencia COCO. Una época no basta para juzgar precisión ni compararlo limpiamente con los modelos LMD aleatorios de los smoke tests.

**Fuentes:** `runs/detect/train/args.yaml`; `runs/detect/train/results.csv`; `runs/detect/train/weights/{best,last}.pt`.

## 8. Reproducción progresiva de LMD_YOLO

### 8.1 Stage 1: LeYOLO-Nano

Se compararon Nano y Small con el neck/head YOLOv8n fijo:

| Modelo | Parámetros | GFLOPs @640 | P3/P4/P5 backbone | Strides |
|---|---:|---:|---|---|
| YOLOv8n 4 clases, control estructural | 3.011.628 | 8,194918 | nativo | 8/16/32 |
| LeYOLO-Nano | 2.337.500 | 6,257459 | 32/64/96 canales | 8/16/32 |
| LeYOLO-Small | 2.840.100 | 7,390566 | 48/88/128 canales | 8/16/32 |

Nano fue elegido por menor distancia normalizada a 1,39 M / 3,9 GFLOPs, no porque coincidiera. Su diferencia de +947.500 parámetros y +2,357459 GFLOPs demuestra que el simple reemplazo del backbone no reproduce las cifras publicadas.

El port conserva bias en Conv2d, la semántica explícita de expansión y un argumento `attention` aceptado pero no usado, conforme al código público LeYOLO según las notas locales.

**Fuentes:** `docs/architecture_decisions.md`; `outputs/stage1_leyolo/structural_comparison.json`; `models/yolov8n_leyolo_{nano,small}_stage1.yaml`; `third_party/ultralytics/nn/modules/leyolo.py`.

### 8.2 Stage 2: MGConv

Se evaluaron sin entrenamiento dos interpretaciones:

| Variante | Grupos main/atajo | Posiciones | Parámetros | GFLOPs | Decisión |
|---|---|---|---:|---:|---|
| A_g1 | 1/1 | 21, 24 | 2.358.364 | 6,283674 | Conservada como experimental |
| B_g2 | 2/1 | 21, 24 | 2.266.204 | 6,165709 | Canónica local |

La rama principal es Conv 3×3 stride 2, BN y GELU; el atajo es Conv 1×1 stride 2 sin agrupación; ambas usan bias. Se preservan P3/P4/P5 de 64×80×80, 128×40×40 y 256×20×20, Detect convencional y strides 8/16/32.

**Fuentes:** `outputs/stage2_mgconv_experimental/comparison.json`; `outputs/stage2_mgconv/structural_validation.json`; `models/yolov8n_leyolo_nano_mgconv_stage2.yaml`; `third_party/ultralytics/nn/modules/mg_conv*.py`.

### 8.3 Stage 3: Detect_DBB

Se construyeron dos hipótesis:

| Variante | DBB | TRAIN parámetros / GFLOPs | DEPLOY parámetros / GFLOPs | Selección |
|---|---:|---:|---:|---|
| A | 6 | 3.073.884 / 8,566170 | 2.265.820 / 6,157107 | Canónica |
| B | 12 | 3.372.636 / 10,282189 | 2.265.436 / 6,148506 | Experimental |

La variante A reemplaza el primer Conv 3×3 de cada torre de regresión `cv2` y clasificación `cv3` en tres escalas. Mantiene el segundo Conv, proyecciones 1×1, `reg_max=16`, DFL, decodificación anchor-free y strides 8/16/32.

Cada DBB TRAIN suma cuatro ramas: 3×3+BN; 1×1+BN; 1×1/identidad+BN/padding→3×3+BN; y 1×1+BN/padding→AvgPool+BN. DEPLOY contiene una sola Conv2d 3×3 con bias seguida de SiLU.

**Fuentes:** `docs/detect_dbb_experimental.md`; `docs/stage3_detect_dbb.md`; `outputs/detect_dbb_experimental/structural_comparison.json`; `third_party/ultralytics/nn/modules/diverse_branch_block.py`.

### 8.4 TRAIN frente a DEPLOY

La conversión carga el `best.pt` Stage 3, usa FP32/eval, hace copia del modelo y aplica `switch_to_deploy()` a seis DBB. El checkpoint TRAIN original conservó SHA-256 `6ae788c05d9bc66ef1a600e0fcd895455a8f115c97797e5b49b48a528b2050a4`; el DEPLOY tiene SHA-256 `07ac5b5ac8e1691739f69d32e8d2e8a17fa6761d2029b7febe9f2427edafc852`.

| Propiedad | TRAIN | DEPLOY |
|---|---:|---:|
| DBB | 6 multirrama | 6 fusionados, una Conv 3×3 cada uno |
| Parámetros | 3.073.884 | 2.265.820 |
| GFLOPs @640 | 8,566170 | 6,157107 |
| Diferencia | — | −808.064 parámetros; −2,409063 GFLOPs |
| Strides/salida | 8/16/32; `[1,8,8400]` | iguales |

La equivalencia estricta CPU FP32 con MKLDNN deshabilitado pasó: tolerancia de bloque `rtol=1e-4, atol=1e-5`; modelo `rtol=1e-3, atol=1e-4`; máxima diferencia de salida decodificada `0,0001220703125`. Con MKLDNN activado, un elemento de `P5_cv3` excedió la tolerancia de bloque, pero el modelo completo siguió dentro de tolerancia. No se relajaron umbrales.

**Fuentes:** `outputs/stage3_detect_dbb/deploy_validation.json`; `backend_diagnostic.json`; `scripts/convert_validate_detect_dbb_stage3.py`; `runs/detect/leyolo_nano_mgconv_dbb_stage3_deploy/weights/best_deploy_fp32.pt`.

## 9. Validaciones iniciales y smoke tests

Todos los smoke locales usaron CPU, una época, 3.082 imágenes train, 385 val, 640 px, batch 4, seed 0, determinismo, pesos aleatorios para las etapas LMD y `split: val`. Stage 1–3 configuraron `optimizer:auto`; las notas indican que Ultralytics eligió AdamW.

| Run | Tiempo s | P | R | mAP50 | mAP50-95 | Interpretación |
|---|---:|---:|---:|---:|---:|---|
| LeYOLO Stage 1 | 1.630,61 | 0,00145 | 0,20185 | 0,00196 | 0,00037 | Pipeline entrenable y serializable |
| MGConv Stage 2 | 1.563,51 | 0,00155885 | 0,23617149 | 0,00357338 | 0,00083303 | Forward/gradientes/reload/val pasan |
| Detect_DBB Stage 3 | 1.418,74 | 0,00165263 | 0,24930300 | 0,00735643 | 0,00100034 | TRAIN multirrama entrenable |

Las cifras no establecen exactitud ni mejora causal: son ejecuciones de una época y no una comparación controlada con repeticiones.

**Fuentes:** `runs/detect/*_smoke/args.yaml`; `results.csv`; `outputs/stage1_leyolo/smoke_training.md`; `outputs/stage2_mgconv/smoke_training.json`; `outputs/stage3_detect_dbb/smoke_training.json`.

## 10. Overfit64

### 10.1 Objetivo y subconjunto

Stage 5 intentó comprobar capacidad real de aprendizaje/memorización del modelo TRAIN, no generalización. Partió del checkpoint Stage 3 de una época y seleccionó 64 imágenes únicas del split train con seed `20260923`: primero 24 con D40, luego 16 con D20, 12 con D10 y 12 con D00, sin reemplazo. Train y val son exactamente el mismo conjunto.

| Clase | Objetos | Imágenes que contienen la clase |
|---|---:|---:|
| D00 | 49 | 31 |
| D10 | 25 | 19 |
| D20 | 34 | 25 |
| D40 | 34 | 24 |
| **Total** | **142** | — |

Se verificaron 64 imágenes y labels, cero fondos/corruptos, clases 0–3 y coordenadas finitas normalizadas.

**Fuentes:** `outputs/stage5_overfit64/manifest.{csv,json}`; `preflight.json`; `data/overfit64`; `scripts/run_stage5_overfit64.py`.

### 10.2 Configuración

| Opción | Valor |
|---|---|
| Modelo | LMD Stage 3 TRAIN, seis DBB multirrama |
| Inicialización | Stage 3 `best.pt`, SHA-256 `6ae788...50a4` |
| Épocas / batch / imgsz | 50 / 8 / 640 |
| Dispositivo | CPU |
| Optimizer | AdamW |
| LR inicial/final factor | 0,001 / 0,1 |
| beta1 (`momentum`) / weight decay | 0,9 / 0,0005 |
| Warmup | 3 épocas; momentum 0,8; bias LR 0,01 |
| Seed | 20260923, deterministic |
| Augmentations | todas las geométricas/fotométricas, mosaic, mixup, cutmix y copy-paste desactivadas |

### 10.3 Evolución y resultado

| Época | box | cls | DFL | P | R | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,22135 | 5,84039 | 4,30346 | 0,00042 | 0,05147 | 0,00012 | 0,00004 |
| 20 | 2,44655 | 3,44002 | 2,95342 | 0,00057 | 0,08373 | 0,00030 | 0,00014 |
| 30 | 1,75463 | 2,54950 | 2,25219 | 0,76001 | 0,28663 | 0,30769 | 0,20366 |
| 40 | 1,35074 | 1,97869 | 1,85179 | 0,81202 | 0,41368 | 0,48435 | 0,34232 |
| 50 | 1,18378 | 1,79883 | 1,71242 | 0,92443 | 0,40901 | 0,51420 | 0,38252 |

Entre épocas 1 y 50, box/cls/DFL cayeron 72,0 % / 69,2 % / 60,2 %. No hubo NaN/Inf. La validación explícita del `best.pt` produjo P 0,933498; R 0,409056; mAP50 0,516547; mAP50-95 0,383061.

| Clase | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| D00 | 0,86585 | 0,38776 | 0,48066 | 0,32630 |
| D10 | 0,88181 | 0,40000 | 0,46503 | 0,26368 |
| D20 | 0,98633 | 0,52941 | 0,69990 | 0,60292 |
| D40 | 1,00000 | 0,31906 | 0,42060 | 0,33935 |

Entrenamiento/validación interna: 2.443,52 s; validación explícita: 12,10 s; total: 2.457,28 s. El `best.pt` conserva los seis DBB TRAIN y tiene SHA-256 `d73761e38d9f90d2a0e79908099898d1e85aa9386ba32eaba4ed829c0b8f28cb`.

**Interpretación correcta:** el modelo aprende de forma coherente y todas las clases producen detecciones, pero no memorizó suficientemente las 64 imágenes al límite de 50 épocas; recall 0,4091 y omisiones visuales hacen el diagnóstico **INCONCLUSIVE** para overfit completo. Estas métricas no estiman generalización.

**Fuentes:** `runs/detect/lmd_stage5_overfit64/results.csv`; `outputs/stage5_overfit64/training_summary.json`; `report.md`; predicciones y mosaicos de la misma carpeta.

## 11. Preparación para on-device y exportación ONNX

### 11.1 Modelo congelado

El artefacto móvil deriva del checkpoint **Stage 3 de una época**, convertido a DEPLOY; no deriva de Overfit64 ni de Stage 6. Esto valida portabilidad técnica, no precisión definitiva.

| Elemento | Valor |
|---|---|
| Checkpoint DEPLOY | `runs/detect/leyolo_nano_mgconv_dbb_stage3_deploy/weights/best_deploy_fp32.pt` |
| SHA-256 PT | `07ac5b5ac8e1691739f69d32e8d2e8a17fa6761d2029b7febe9f2427edafc852` |
| Tamaño PT | 9.453.763 bytes |
| Parámetros / GFLOPs | 2.265.820 / 6,157107 @640 |
| Precisión | FP32 |

### 11.2 ONNX

| Propiedad | Valor |
|---|---|
| Archivo | `outputs/stage4_mobile_export_audit/onnx_fp32_640/best_deploy_fp32_source_copy.onnx` |
| SHA-256 | `a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8` |
| Tamaño | 9.214.905 bytes |
| Configuración | FP32, batch 1, 640 fijo, `dynamic=False`, `simplify=False`, `nms=False` |
| ONNX / ORT PC | 1.23.0 / 1.30.0 |
| Opset / IR | 18 / 8 |
| Entrada / salida | `images [1,3,640,640]` / `output0 [1,8,8400]` |
| Grafo | 284 nodos, todos `ai.onnx`, sin custom domains |

La comparación PyTorch–ORT PC, sobre `China_Drone_000001.jpg` de `val`, dio máximo absoluto 0,0011901855 y media 1,7510e-05; pasó ambos `allclose` documentados. El `.pt` original no cambió.

**Fuentes:** `outputs/stage4_mobile_export_audit/deploy_structure.json`; `onnx_fp32_640_validation.json`; `docs/stage4_mobile_export_audit.md`; `scripts/audit_mobile_export_stage4.py`.

## 12. Preprocesamiento canónico PC–Android

### 12.1 Problema inicial

El primer benchmark usó `cv2.resize(INTER_LINEAR)` en PC y `Bitmap.createScaledBitmap(filter=true)` en Android. Aunque ambos son bilineales, difirieron en coordenadas/coefs/redondeo: hasta 9 niveles RGB; 22.931 elementos del tensor fallaron la tolerancia y 17.022 salidas ONNX quedaron fuera. Como la imagen era 512×512 y el destino 640×640, no había padding; la causa fue el resizer, no letterbox.

**Fuentes:** `mobile_benchmark/results/cpu_xnnpack_1790064594356.json`; `docs/canonical_mobile_preprocessing.md`.

### 12.2 Contrato canónico

1. Matriz ya orientada; sin EXIF implícito divergente.
2. RGB8 opaco sRGB.
3. LetterBox 640×640: `auto=false`, `scale_fill=false`, `scaleup=true`, centrado, padding RGB 114.
4. Ratio `min(640/h,640/w)`; tamaño redondeado al par más cercano; sobrante impar a derecha/abajo.
5. Bilinear half-pixel con pesos racionales enteros, clamp de borde y redondeo half-up a RGB8.
6. HWC RGB→NCHW, float32 y división por `float32(255.0)`.
7. Forma `[1,3,640,640]`.

La misma lógica reside en `scripts/generate_canonical_mobile_golden.py` y `mobile_benchmark/app/src/main/java/com/lmdyolo/benchmark/CanonicalPreprocessor.kt`.

### 12.3 Validación

En `preprocessing_validation_1790066184495.json`:

- estado `PASS_CANONICAL_EXACT`;
- 786.432/786.432 valores RGB8 PC/Android idénticos para esta imagen;
- 1.228.800/1.228.800 valores de tensor idénticos bit a bit;
- salida de 67.200 valores: máximo absoluto 0,001159668, media 1,61414e-05, cero fuera de `rtol=1e-3, atol=1e-3`;
- cero candidatos/detecciones con confianza 0,25 en ambos lados;
- letterbox+tensor en una medición debug: 26,416 ms.

La identidad del decoder JPEG solo se observó en esa imagen/dispositivo; no es garantía universal.

**Fuentes:** `docs/canonical_mobile_preprocessing.md`; `mobile_benchmark/results/preprocessing_validation_1790066184495.json`; assets `canonical_*`.

## 13. Implementación Android

Se creó un harness Android nativo aislado, no la aplicación final React Native/Expo. Empaqueta el ONNX, golden tensors, imagen y metadata; verifica hashes; ejecuta ORT; mide decode, letterbox, tensor, inferencia, postproceso, NMS y extremo a extremo; captura memoria y estado térmico; y perfila proveedores.

| Componente | Archivo |
|---|---|
| Ejecutor/benchmark | `mobile_benchmark/app/src/main/java/com/lmdyolo/benchmark/BenchmarkRunner.kt` |
| Preprocesador | `.../CanonicalPreprocessor.kt` |
| UI mínima | `.../MainActivity.kt` |
| Dependencias Android | `mobile_benchmark/app/build.gradle.kts` |
| Procedimiento | `mobile_benchmark/README.md` |
| Resultados | `mobile_benchmark/results/*.json` |

El runtime observado en el dispositivo fue ONNX Runtime Android 1.24.3. El golden PC fue generado con ORT 1.30.0; la salida pasó la tolerancia fijada.

## 14. Prueba NNAPI

La prueba corta en Samsung Galaxy S23 SM-S911B, Android 16, SoC QTI SM8550, FP32 y el ONNX congelado terminó **ABORTED** durante `OrtEnvironment.createSession`. No hubo inferencia, warmup, benchmark 10+100, fallback medido ni perfil completo.

ORT informó candidatos de partición (247/249 y 255/257 nodos soportados en vistas internas), pero eso no demuestra ejecución. El error fatal exacto fue:

```text
op_builder_helpers.cc:144 AddNnapiSplit count [0] does not evenly divide dimension 3 [128]
```

El ONNX contiene cuatro `Split` válidos sobre canales: `[64,64]`, `[32,32]`, `[64,64]`, `[128,128]`. La documentación infiere que `count [0]` proviene de traducción/construcción del NNAPI EP bajo mapeo NHWC, no de una división inválida en el ONNX; no es una conclusión de perfil ejecutado. No se reescribió el modelo, cambió opset/runtime ni relajó precisión.

**Fuentes:** `mobile_benchmark/results/nnapi_validation_1790067014298.json`; `docs/nnapi_short_validation_failure.md`; `scripts/inspect_onnx_split_nodes.py`.

## 15. Benchmark definitivo CPU/XNNPACK en Galaxy S23

### 15.1 Condiciones

- Samsung Galaxy S23 SM-S911B (`dm1q`), QTI SM8550, 8 procesadores, Android 16/API 36.
- ORT Android 1.24.3; XNNPACK explícito con 4 threads y ORT intra-op 1; ejecución mixta XNNPACK+CPU.
- FP32, input `[1,3,640,640]`, output `[1,8,8400]`.
- 10 warmups y 100 ejecuciones medidas, desconectado, batería 100 %, 28,7 °C antes/después; sensor de batería, no del SoC.
- Resultado canónico: `cpu_xnnpack_1790067825791.json`, SHA-256 `ec8f75953cd276c825457e36b973effd6b8b03c4b5e1ce386275cd6dbfacbc98`.

### 15.2 Latencia

| Etapa | Media ms | P50 | P90 | P95 | Min–max |
|---|---:|---:|---:|---:|---:|
| Decode JPEG | 3,454 | 3,484 | 3,596 | 3,608 | 2,945–4,128 |
| Letterbox | 28,439 | 28,289 | 29,410 | 29,804 | 27,499–30,560 |
| RGB/NCHW | 4,967 | 4,946 | 5,089 | 5,165 | 4,632–6,749 |
| Preparación total | 36,860 | 36,768 | 37,800 | 38,103 | 35,715–39,022 |
| Inferencia | 76,810 | 76,741 | 77,698 | 78,112 | 75,363–79,055 |
| Postproceso | 0,064 | 0,064 | 0,065 | 0,066 | 0,062–0,104 |
| NMS | 0,032 | 0,031 | 0,034 | 0,036 | 0,022–0,051 |
| End-to-end | 113,772 | 113,634 | 115,022 | 115,635 | 111,734–116,696 |

FPS equivalentes: 13,019 solo inferencia y 8,790 end-to-end. Carga de sesión: 59,060 ms. La imagen no generó candidatos a confianza 0,25, por lo que el coste de postproceso/NMS no representa escenas con numerosas detecciones.

### 15.3 Estabilidad, memoria y proveedores

- Inferencia primeras/últimas 20: 76,399/77,251 ms, degradación +1,116 %.
- End-to-end primeras/últimas 20: 113,606/113,843 ms, +0,208 %.
- CV de inferencia 1,02 %; end-to-end 0,84 %; no se observó ralentización sostenida >15 %.
- Estado térmico Android `NONE` antes, tras warmup, cada 10 runs y al final.
- PSS: 117.183 KiB antes de sesión; 155.484 KiB después; 218.103–261.587 KiB durante runs; 256.679 KiB al terminar antes del perfil.
- Heap nativo run 10→100: +1.322.672 bytes; no muestra fuga acelerada en 100 runs, pero no es una prueba soak.
- Perfil separado: 81 eventos/nodos XNNPACK (Conv, MaxPool) y 112 CPU (Add, Concat, Div, Gelu, Mul, QuickGelu, Reshape, Resize, Sigmoid, Softmax, Split, Sub, Transpose). No deben interpretarse como porcentajes del grafo o latencia.

Conclusión limitada: el ONNX FP32 es técnicamente viable con CPU/XNNPACK en **este Galaxy S23**, con ~13 FPS de inferencia y ~8,8 FPS del pipeline debug canónico. No valida todos los smartphones, una app final, video sostenido, energía del SoC ni rendimiento release.

**Fuentes:** `docs/cpu_xnnpack_canonical_baseline.md`; `mobile_benchmark/results/cpu_xnnpack_1790067825791.json`.

## 16. Stage 6: comparación controlada YOLOv8n vs LMD_YOLO

### 16.1 Evidencia recuperada y alcance

La segunda auditoría localizó la corrida completa en `results_stage6/results-20260923T224020Z-1-001/results/`: 129 archivos y 195.296.616 bytes. Contiene `preflight.json`, `smoke_gate.json`, configuración efectiva, CSV, `results.png`, curvas PR/F1/P/R, matrices de confusión normalizadas y sin normalizar, lotes train/val, resúmenes, validaciones explícitas del mejor checkpoint, pesos `best`/`last` y checkpoints periódicos, además de `comparison.json` y `learning_curves.png`. No contiene notebook ejecutado, log textual completo de consola ni un manifest propio de esa carpeta; los manifests del paquete/dataset permanecen en `colab_stage6/manifests/`. Estas ausencias no impiden auditar los resultados presentes.

### 16.2 Metodología efectiva y comparabilidad

Ambos modelos se construyeron desde YAML y partieron desde cero con seed 42. Esto evita favorecer a YOLOv8n con COCO o a LMD con un checkpoint previo no equivalente. La igualdad de hiperparámetros está confirmada por ambos `args.yaml`:

| Variable común | Valor efectivo |
|---|---|
| Épocas / imagen / batch | 20 / 640 / 16 |
| Inicialización | desde cero en ambos (`pretrained=false`) |
| Optimizer | SGD |
| LR `lr0` / `lrf` | 0,01 / 0,01 |
| Momentum / weight decay | 0,937 / 0,0005 |
| Warmup | 3 épocas; momentum 0,8; bias LR 0,1 |
| AMP / deterministic | true / true |
| Workers / patience | 2 / 100 |
| Seed | 42 |
| Datos | 3.082 train y 385 val; 386 TEST reservados |
| Augmentation | HSV 0,015/0,7/0,4; translate 0,1; scale 0,5; flip horizontal 0,5; mosaic 1,0 y `close_mosaic=10`; restantes 0 |

La comparación es controlada para esta corrida, pero una sola seed y 20 épocas no cuantifican variabilidad ni superioridad general.

### 16.3 Preflight efectivo

`preflight.json` terminó `PASS` con Tesla T4 de 15.637.086.208 bytes (~14,56 GiB), Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 y Ultralytics 8.4.153 cargado desde `/content/colab_stage6/vendor/ultralytics/__init__.py`. El sufijo `cu128` documenta una build CUDA 12.8; el JSON no registra por separado versión de driver o toolkit. Verificó 7.706 archivos del dataset con hash completo y conteos 3.082/385/386.

| Modelo | Estructura | Parámetros | MACs @640 | GFLOPs = 2×MACs | Forward/backward T4 batch 2 | Pico |
|---|---|---:|---:|---:|---|---:|
| YOLOv8n | strides 8/16/32 | 3.011.628 | 4.097.459.200 | 8,1949184 | PASS | 395,19 MiB |
| LMD_YOLO TRAIN | LeYOLO-Nano; MGConv 21/24, `groups=2`; 6 DBB; strides 8/16/32 | 3.073.884 | 4.283.084.800 | 8,5661696 | PASS | 608,43 MiB |

LMD tiene 62.256 parámetros más (+2,07 %) y 0,371252 GFLOPs más (+4,53 %) que el baseline de esta reproducción. Esto difiere del objetivo de reducción publicado porque el modelo local preserva decisiones propias de neck/head y no replica exactamente el diseño de los autores.

### 16.4 Smoke gate efectivo

El smoke usó una época, `fraction=0.02`, imagen 640, batch 16, seed 42, SGD y pesos desde cero. `smoke_gate.json` terminó `PASS` para ambos, sin valores no finitos:

| Modelo | Tiempo wall-clock | Tiempo acumulado del CSV | Pico GPU |
|---|---:|---:|---:|
| YOLOv8n | 34,982 s | 11,405 s | 1.909,84 MiB |
| LMD_YOLO | 62,432 s | 39,569 s | 2.673,05 MiB |

El wall-clock incluye preparación y overhead del proceso; el campo temporal del CSV mide la época acumulada. Para duración del gate se usa el primero. El 2 % es un subconjunto funcional, no una evaluación de precisión.

### 16.5 Incidente THOP y corrección

La primera versión invocaba `ultralytics.utils.torch_utils.get_flops`, que devuelve 0,0 silenciosamente si falta `thop` o se produce una excepción. El requirements inicial de Colab no instalaba la dependencia indirecta presente localmente; el 0,0 fue un fallback de instrumentación, no una propiedad de LeYOLO, MGConv, DetectDBB, CUDA ni PyTorch.

La corrección fijó `ultralytics-thop==2.1.6`, perfiló directamente una copia CPU FP32/eval con tensor real 1×3×640×640, cotejó parámetros THOP con PyTorch y convirtió ausencia, excepción o MAC inválido en fallo explícito. `preflight.json` confirma método directo, versión 2.1.6 y los valores 4.097.459.200/4.283.084.800 MACs. No se reparametrizó DBB para ese perfil.

### 16.6 Resultados globales de validación

`comparison.json` marca la revisión como `READY_FOR_TECHNICAL_REVIEW` y ambos entrenamientos como `COMPLETE`, con 20 épocas. Se reportan las métricas de la validación explícita del `best.pt`, no las cifras ligeramente distintas devueltas al cerrar `train()`:

| Modelo | Parámetros | GFLOPs | Best epoch | P | R | mAP50 | mAP50-95 | Entrenamiento | s/época | Pico GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.011.628 | 8,194918 | 20 | 0,570209 | 0,420032 | 0,443694 | 0,216069 | 1.057,830 s (17:37,83) | 52,892 | 1.933,79 MiB |
| LMD_YOLO | 3.073.884 | 8,566170 | 20 | 0,609036 | 0,411853 | 0,461682 | 0,230440 | 1.246,227 s (20:46,23) | 62,311 | 2.597,53 MiB |
| **LMD − YOLO** | **+62.256 (+2,07 %)** | **+0,371252 (+4,53 %)** | — | **+0,038827** | **−0,008179** | **+0,017988** | **+0,014371** | **+188,396 s (+17,81 %)** | **+9,420 (+17,81 %)** | **+663,75 MiB (+34,32 %)** |

En esta corrida, LMD mejora precisión, mAP50 y mAP50-95, con recall global ligeramente menor y mayor tiempo/memoria. Las mejoras equivalen a +1,80 y +1,44 puntos porcentuales absolutos de mAP; no justifican por sí solas una afirmación de superioridad general.

Hashes SHA-256 de los mejores checkpoints: YOLOv8n `34439ac964e8ca81cf987d9046791c68d3647d2b1662e1be41da2e73fb2f1f6b`; LMD `1d7b292dc93923bf9be9c767b304a43bb025a859848baddadbf0e65f0e7ab24b`.

### 16.7 Resultados por clase

| Clase | Modelo | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| D00 | YOLO / LMD | 0,542109 / 0,613561 | 0,500000 / 0,438462 | 0,497684 / 0,495070 | 0,233416 / 0,234926 |
| D10 | YOLO / LMD | 0,593816 / 0,606014 | 0,556113 / 0,498039 | 0,594611 / 0,539483 | 0,261356 / 0,258035 |
| D20 | YOLO / LMD | 0,697215 / 0,852645 | 0,329897 / 0,357969 | 0,385575 / 0,468452 | 0,215586 / 0,261771 |
| D40 | YOLO / LMD | 0,447695 / 0,363924 | 0,294118 / 0,352941 | 0,296905 / 0,343721 | 0,153915 / 0,167028 |

LMD aporta su ganancia principal en D20 y mejora recall/mAP de D40; pierde mAP50 en D00 y, sobre todo, D10. D40 sigue siendo la clase más débil en mAP50-95, coherente con sus solo 17 objetos en `val`; cualquier conclusión por clase necesita cautela.

### 16.8 Curvas, estabilidad y convergencia

Los 20 registros de ambos CSV son finitos, sin NaN/Inf. Las losses de entrenamiento bajan de forma sostenida: YOLO box/cls/dfl 4,3023/4,5615/4,0737→1,7871/1,9670/1,9149; LMD 4,1848/4,4714/3,9475→1,7876/1,9882/1,9292. Las losses de validación también descienden globalmente; hubo picos tempranos de `val/cls_loss` —YOLO época 3 y LMD época 2— seguidos de recuperación, no divergencia.

Precision, recall y mAP muestran oscilación inicial normal y tendencia ascendente. De época 19 a 20, YOLO mAP50 0,43254→0,44338 y mAP50-95 0,21535→0,21566; LMD 0,44610→0,46155 y 0,21818→0,22805. Ambos `best_epoch` son 20. No hay evidencia de colapso ni sobreajuste claro, pero tampoco de convergencia o plateau: las métricas todavía mejoran en el borde del presupuesto. Veinte épocas son insuficientes para atribuir el límite observado a la arquitectura.

### 16.9 Estado de TEST

**El conjunto TEST permaneció reservado y no fue utilizado durante Stage 6.** `args.yaml` usa `split: val`, `comparison.json` registra `test_used: false` y el código solo recorre TEST para integridad/hashes. No se ejecutó evaluación final sobre sus 386 imágenes.

**Fuentes Stage 6:** `results_stage6/results-20260923T224020Z-1-001/results/{preflight.json,smoke_gate.json,runtime_data.yaml,comparison/comparison.json,comparison/learning_curves.png,training/*/args.yaml,training/*/results.csv,training/*/stage6_summary.json,training/*/weights/best.pt,best_validation/*}`; `colab_stage6/FLOPS_PROFILING.md`; `colab_stage6/scripts/stage6.py`.

## 17. Decisión experimental posterior a Stage 6

La decisión respaldada por la corrida es **continuar con LMD_YOLO como candidato principal**, sin descartar YOLOv8n como control. El fundamento es concreto: LMD gana +0,017988 mAP50 y +0,014371 mAP50-95, con mejora especialmente fuerte en D20, y mantiene entrenamiento estable. El coste también es concreto: +17,81 % de tiempo por época, +34,32 % de pico GPU, +2,07 % de parámetros y +4,53 % de GFLOPs frente al baseline local, además de −0,008179 de recall global.

Por tanto, “continuar” significa ampliar la comparación —más épocas y seeds, luego TEST bloqueado y benchmark móvil del checkpoint resultante—, no declarar que LMD ya es superior ni que reproduce la eficiencia del paper. Los artefactos recuperados justifican la elección técnica; la autoría/fecha de la decisión humana original no está registrada en un acta separada.

## 18. Evidencia acumulada y conclusiones calibradas

1. **Funciona técnicamente:** sí. La arquitectura se construye, hace forward/backward, entrena, serializa, recarga, valida, reparametriza y exporta a ONNX según artefactos locales.
2. **Aprende correctamente:** sí, dentro de la evidencia disponible. Overfit64 y Stage 6 muestran descenso coherente de losses, métricas crecientes y ausencia de NaN/Inf; Overfit64 no logró memorización completa.
3. **Es viable en el dispositivo probado:** sí, como ONNX FP32 con CPU/XNNPACK en un Galaxy S23, con las latencias medidas. Esto no implica viabilidad en todos los teléfonos ni que la aplicación final alcance esos números.
4. **NNAPI:** no validado y, con ORT Android 1.24.3 + ONNX opset 18 congelado, descartado para ese benchmark porque no se crea la sesión.
5. **Presentó mejores mAP en una comparación controlada:** sí. En Stage 6 LMD logró +0,017988 mAP50 y +0,014371 mAP50-95 frente a YOLOv8n, con precisión mayor, recall algo menor y costes de cómputo superiores.
6. **Es superior de forma general:** no demostrado. Requiere más épocas, múltiples seeds, evaluación TEST una sola vez al final, intervalos/variabilidad y mediciones del checkpoint comparado en hardware objetivo.
7. **Precisión definitiva:** no disponible. El artefacto móvil procede de un smoke de una época.

### 18.1 Matriz Paper vs implementación vs evidencia propia

| Tema | Paper | Implementación local | Evidencia propia / lectura correcta |
|---|---|---|---|
| Backbone | LeYOLO ligero; detalle incompleto (pp. 4–5). | LeYOLO-Nano elegido por proximidad y validado. | Funciona, pero no es réplica demostrada; Stage 6 local conserva 3,074 M parámetros. |
| `mn_conv` | Conv–BN–SiLU, grupos posibles (p. 5). | Port local con decisiones explícitas. | El paper no fija grupos efectivos ni YAML. |
| `mg_conv` | GELU + residual en neck (pp. 5–6, Fig. 7). | Dos módulos, índices 21/24, `groups=2`, shortcut propio. | Ubicación exacta, grupos, kernel y shortcut son decisiones del proyecto. |
| Detect_DBB | Head multirrama reparametrizable (pp. 6–7, Fig. 8). | Seis DBB en `cv2`/`cv3` de P3/P4/P5; cuatro ramas del DBB oficial. | TRAIN→DEPLOY está validado; cantidad/posiciones y rama promedio no están fijadas inequívocamente por el paper. |
| Modelo completo | LeYOLO+MGConv+DBB, 1,70 M, 3,8 G, 63 FPS (Tablas 5–6). | TRAIN local: 3,073884 M, 8,566170 GFLOPs. | Reproducción funcional, no réplica de eficiencia. |
| Entrenamiento | 200 épocas, batch 64, A100 40 GB, annealing cosenoidal (Tabla 1–2). | Stage 6: 20 épocas, batch 16, T4, `cos_lr=false`, scratch. | Comparación interna válida; no reproduce protocolo del paper ni demuestra convergencia. |
| Métricas | LMD mAP50 0,893 y mAP50-95 0,556 (Tabla 6). | Validación Stage 6: 0,461682 y 0,230440. | No son comparables directamente por protocolo, presupuesto e implementación distintos. |
| Móvil | Orientación/aptitud declarada para móvil/edge. | ONNX FP32 medido en Galaxy S23. | La validación física es aporte propio: ~13,0 FPS inferencia y ~8,8 FPS end-to-end; no procede del paper. |
| TEST | Evaluación publicada sobre su protocolo RDD2022. | TEST local de 386 imágenes bloqueado. | Stage 6 no lo usó; precisión final todavía pendiente. |

## 19. Limitaciones y contradicciones

| Tema | Evidencia en tensión | Tratamiento |
|---|---|---|
| Paper MGConv | Tablas citadas: 1,34 M; prosa citada: 1,36 M. | Se registran ambas; no se elige silenciosamente. |
| Baseline del paper | Sección IV.A dice “YOLOv8s”, pero arquitectura, tablas y comparaciones usan YOLOv8n. | Se adopta YOLOv8n como baseline por predominio de evidencia, dejando constancia de la errata/contradicción. |
| Tabla 5 frente a prosa | Tabla: G1 61 FPS, G2 1,34 M/60 FPS, G4 P 0,823, R 0,830, mAP50 0,893, mAP50-95 0,556, 63 FPS. La prosa asociada da G1 52 FPS, G2 1,36 M/50 FPS y G4 P 0,893, R 0,771, mAP50 0,855, mAP50-95 0,516, 45 FPS. | Se priorizan los valores tabulados para citar resultados y se conserva la inconsistencia como limitación editorial. |
| FPS final del paper | Resumen, Tabla 5, Tabla 6 y conclusión indican 63 FPS; un párrafo de discusión indica 45 FPS. | No se usa ninguna de las dos cifras como expectativa del teléfono; el protocolo FPS no se describe con suficiente detalle y el entorno publicado es A100/Xeon. |
| Fidelidad LMD | Paper no publica YAML/canales/weights; implementación toma decisiones propias. | Se denomina reproducción funcional, no réplica exacta. |
| DBB promedio | Ecuación del paper sugiere AvgPool→1×1; DBB oficial adaptado usa 1×1→AvgPool y además una rama 1×1 independiente. | Se siguió el código oficial y se documentó la diferencia; no se atribuye esa topología exacta a los autores. |
| Preprocesamiento Android | Primer benchmark: JPEG/tensor no equivalentes; validación posterior canónica: equivalencia exacta. | El resultado posterior `1790066184495` es canónico porque implementa el contrato bloqueado y precede al benchmark definitivo. |
| CPU A frente a B | Preproceso antiguo ~7,24 ms; canónico ~36,86 ms; inferencia casi igual. | No comparar end-to-end como cambio de modelo. |
| TRAIN vs DEPLOY | Menos parámetros/FLOPs, pero PT DEPLOY FP32 puede ser mayor que checkpoint TRAIN guardado en FP16. | Tamaño de archivo no se usa como proxy de complejidad. |
| THOP Stage 6 | Primer profiling dio 0,0; medición corregida da valores no nulos. | 0,0 fue fallback por dependencia ausente, no resultado arquitectónico. |
| THOP vs complejidad publicada | Stage 6 local da 3,074 M/8,566 G; paper da 1,70 M/3,8 G. | No se corrige retrospectivamente el modelo: la diferencia es evidencia de no equivalencia arquitectónica. |
| “Decisión de continuar” | Stage 6 favorece a LMD en mAP, pero no existe acta independiente. | Se documenta como decisión técnica razonada, no como hecho administrativo ni conclusión definitiva. |

Otras limitaciones:

- el paper primario está fuera de `C:\TP1`; falta copia estable y hash dentro del proyecto;
- no hay historial Git ni logs completos;
- las fechas son marcas de archivo;
- Overfit64 parte de un checkpoint ya expuesto al dataset;
- el benchmark móvil usa una imagen sin detecciones a umbral 0,25;
- solo se ensayó un modelo de teléfono;
- APK debug y preprocesador canónico no optimizado;
- no hay prueba prolongada de energía, soak, cámara o aplicación integrada;
- no hay cuantización FP16/INT8 ni rutas iOS/LiteRT/Core ML/NCNN verificadas;
- TEST aún no se ha evaluado.
- Stage 6 usa una sola seed, 20 épocas y termina con `best_epoch=20`; no demuestra convergencia ni variabilidad estadística.

## 20. Estado actual

| Estado | Elemento | Fundamento |
|---|---|---|
| **VALIDADO** | Dataset convertido y split reproducible | 3.853 pares, seed 42, manifest/hashes y conteos coherentes. |
| **VALIDADO** | Reproducción funcional LeYOLO-Nano | Estructura, forward y smoke. |
| **VALIDADO** | MGConv local | Dos capas, `groups=2`, estructura y smoke. |
| **VALIDADO** | Detect_DBB local TRAIN | Seis DBB, entrenamiento y reload. |
| **VALIDADO** | Conversión TRAIN→DEPLOY | Equivalencia y preservación del TRAIN comprobadas. |
| **VALIDADO** | Exportación ONNX FP32 | Checker, ORT PC y comparación numérica PASS. |
| **VALIDADO** | Preprocesamiento canónico en el caso probado | Tensor bit-exacto PC/Android. |
| **VALIDADO** | Inferencia on-device CPU/XNNPACK | Galaxy S23, 10+100 runs, estabilidad y memoria. |
| **VALIDADO** | Estabilidad térmica en el benchmark corto | Estado Android `NONE` durante 100 runs; no equivale a prueba prolongada. |
| **VALIDADO** | Capacidad de aprendizaje | Overfit64 mejora losses/métricas sin NaN, aunque no memoriza por completo. |
| **VALIDADO** | Overfit64 como prueba diagnóstica | Entrena y mejora sobre 64 ejemplos; no es métrica de generalización. |
| **VALIDADO** | Preflight y smoke Stage 6 | Ambos modelos pasan en T4; hashes, arquitectura, forward/backward y finitud comprobados. |
| **VALIDADO** | Comparación controlada Stage 6 | Misma seed/configuración, 20 épocas desde cero; LMD mejora ambos mAP con mayor coste. |
| **VALIDADO** | TEST reservado en Stage 6 | `split: val`, `test_used:false`; TEST solo se hashea. |
| **PENDIENTE** | Evaluación final TEST | No hay ejecución TEST. |
| **PENDIENTE** | Entrenamiento definitivo | No hay run final documentado. |
| **PENDIENTE** | Exportación y validación on-device del modelo definitivo | Los artefactos móviles actuales derivan de Stage 3 de una época. |
| **PENDIENTE** | Precisión definitiva/generalización | Depende de entrenamiento final y TEST. |
| **PENDIENTE** | Integración con aplicación móvil | Solo existe harness Android. |
| **PENDIENTE** | Integración backend y plataforma web | No hay artefactos que demuestren estas integraciones. |
| **PENDIENTE** | Validación en gama media/más dispositivos/iOS | Solo Galaxy S23. |
| **NO VALIDADO** | Réplica exacta del paper | Detalles esenciales no publicados; parámetros/GFLOPs locales difieren ampliamente. |
| **NO VALIDADO** | Superioridad general de LMD | Falta evidencia multiseed, convergencia y TEST; Stage 6 solo valida una comparación acotada. |
| **NO VALIDADO** | Rendimiento móvil afirmado por el paper | El paper no incluye teléfono físico; la prueba Galaxy S23 es propia y usa otro checkpoint. |
| **NO VALIDADO** | Rendimiento con escenas de muchas detecciones | Golden móvil tiene cero candidatos. |
| **NO VALIDADO** | FP16/INT8, LiteRT, Core ML, ExecuTorch, NCNN, MNN | Solo auditoría estática o recomendación. |
| **DESCARTADO en configuración probada** | NNAPI ORT 1.24.3 + ONNX opset 18 congelado | Falla en creación de sesión `AddNnapiSplit`. |
| **DESCARTADO como conclusión** | Métricas smoke como precisión final | Una época es solo validación funcional. |

## 21. Trabajo pendiente priorizado

La siguiente etapa es el **entrenamiento definitivo de LMD_YOLO**. Antes de iniciarlo deben fijarse por escrito presupuesto de épocas/criterio de parada, seeds, mAP50-95 principal, recall mínimo y límites de latencia/memoria; Stage 6 terminó en época 20 sin plateau. El orden futuro es:

1. Entrenar definitivamente LMD_YOLO, idealmente con repetición de seeds o una justificación explícita si solo se usa una.
2. Seleccionar y congelar el mejor checkpoint exclusivamente con TRAIN/VAL y el criterio predefinido.
3. Ejecutar una única evaluación final sobre las 386 imágenes de TEST, después del congelamiento.
4. Analizar resultados globales y por clase, errores, matrices de confusión y variabilidad disponible.
5. Convertir TRAIN→DEPLOY y volver a validar equivalencia numérica sin modificar el checkpoint fuente.
6. Exportar el modelo definitivo —por ejemplo ONNX— y regenerar goldens/versiones/hashes.
7. Repetir validación on-device en Galaxy S23 y al menos un gama media, con detecciones reales, build release, video sostenido, temperatura/energía y NMS representativo.
8. Integrar el modelo congelado con la aplicación móvil y validar el pipeline funcional completo.
9. Integrar backend y plataforma web, definiendo contratos, seguridad, telemetría y versionado del modelo.

En paralelo documental, preservar la carpeta Stage 6 con manifest/hash, recuperar notebook ejecutado/log si existen e incorporar una copia estable del PDF con DOI y SHA-256. Investigar NNAPI debe hacerse en una rama separada, con exactitud validada antes de medir rendimiento.

## 22. Inventario de artefactos y evidencias

### 22.1 Datos y preparación

- `data/raw/China_Drone/train`, `data/raw/China_MotorBike/train`: fuente VOC.
- `scripts/check_classes.py`: recuento previo.
- `scripts/convert_voc_to_yolo.py`: filtro/conversión.
- `scripts/split_dataset.py`: split seed 42.
- `data/data.yaml`, `data/dataset`: dataset canónico.
- `colab_stage6/manifests/dataset_manifest.json`: hashes y digest combinado.

### 22.2 Arquitectura y pruebas

- `models/*.yaml`: candidatos y modelos canónicos Stages 1–3.
- `third_party/ultralytics/nn/modules/{leyolo,mg_conv,detect_dbb,diverse_branch_block}.py`: implementación.
- `tests/test_leyolo_stage1.py`, `tests/test_mgconv_experimental.py`: tests presentes.
- `scripts/validate_*`, `compare_*`, `train_*`: construcción, auditoría y smoke.
- `outputs/stage1_leyolo`, `stage2_mgconv*`, `detect_dbb_experimental`, `stage3_detect_dbb`: resultados estructurales.
- `runs/detect/*_smoke`: CSV, plots y checkpoints.

### 22.3 Overfit64

- `scripts/run_stage5_overfit64.py`.
- `data/overfit64`.
- `outputs/stage5_overfit64/{manifest.csv,manifest.json,preflight.json,training_summary.json,report.md}`.
- `runs/detect/lmd_stage5_overfit64/results.csv` y `weights/`.
- `outputs/stage5_overfit64/predictions*`.

### 22.4 DEPLOY, ONNX y móvil

- `scripts/convert_validate_detect_dbb_stage3.py`.
- `runs/detect/leyolo_nano_mgconv_dbb_stage3_deploy/weights/best_deploy_fp32.pt`.
- `scripts/audit_mobile_export_stage4.py`.
- `outputs/stage4_mobile_export_audit/*`.
- `scripts/generate_mobile_benchmark_golden.py`, `generate_canonical_mobile_golden.py`, `inspect_onnx_split_nodes.py`.
- `docs/stage4_mobile_export_audit.md`, `canonical_mobile_preprocessing.md`, `nnapi_short_validation_failure.md`, `cpu_xnnpack_canonical_baseline.md`.
- `mobile_benchmark/app/src/main/*`, assets y `mobile_benchmark/results/*.json`.

### 22.5 Stage 6

- `colab_stage6.zip` y carpeta `colab_stage6/`.
- `stage6_colab.ipynb`, `scripts/stage6.py`, `configs/comparison.yaml`.
- `AUDIT.md`, `FLOPS_PROFILING.md`, `README.md`, `requirements-colab.txt`.
- `models/{yolov8n,lmd_yolo_stage3}.yaml`, `reference_weights/`, `vendor/ultralytics/`.
- `manifests/{dataset_manifest,package_manifest}.json`.
- `results_stage6/results-20260923T224020Z-1-001/results/preflight.json` y `smoke_gate.json`.
- `.../results/smoke/{yolov8n,lmd_yolo}/`: args, CSV, plots, imágenes y pesos.
- `.../results/training/{yolov8n,lmd_yolo}/`: args, CSV, resumen, plots, imágenes y checkpoints.
- `.../results/best_validation/{yolov8n,lmd_yolo}/`: validación explícita de mejores pesos.
- `.../results/comparison/{comparison.json,learning_curves.png}`.
- **Ausentes en la carpeta recuperada:** notebook ejecutado y log textual completo de consola.

## 23. Puntos que requieren confirmación humana

1. Si `full_text (11).pdf` es la versión bibliográfica definitiva que debe archivarse en `paper/` y quién autoriza esa copia.
2. Si hubo otras corridas Stage 6 parciales o completas en Drive; no deben mezclarse con `results-20260923T224020Z-1-001`.
3. Quién tomó y en qué fecha la decisión humana de continuar con LMD; la evidencia técnica la respalda, pero no existe acta.
4. Si existieron entrenamientos, benchmarks o evaluaciones TEST fuera de `C:\TP1`; la evidencia auditada mantiene TEST intacto.
5. Si el Galaxy S23 probado es el dispositivo objetivo de producción o solo un dispositivo disponible.
6. Cuál será el criterio de parada y selección del experimento definitivo, dado que ambos modelos aún mejoraban en época 20.

## 24. Dictamen sobre suficiencia documental

La documentación es **suficiente para reconstruir de forma confiable** la preparación del dataset, el baseline inicial, la evolución arquitectónica Stages 1–3, smoke tests, Overfit64, conversión TRAIN→DEPLOY, ONNX, preprocesamiento, pruebas Android y la comparación controlada Stage 6. El PDF permite además separar con páginas/tablas qué procede del paper y qué fue decisión propia.

También es suficiente para sostener una decisión acotada: continuar investigando LMD_YOLO porque mejoró ambos mAP en la corrida Stage 6, aceptando un coste mayor. No es suficiente para afirmar réplica exacta, convergencia, superioridad general, precisión final ni ventaja móvil del checkpoint Stage 6. Faltan repetición multiseed, entrenamiento hasta plateau, evaluación única de TEST, exportación/benchmark del checkpoint final, log/notebook ejecutado y confirmación de cualquier actividad fuera del workspace.
