# Etapa 3 canónica de nuestra reproducción de LMD_YOLO

Esta etapa reúne LeYOLO-Nano, `MGConv(groups=2)` en las capas 21 y 24 del
neck y `DetectDBB` con seis bloques reparametrizables. **No afirmamos que el
LMD_YOLO publicado utilice exactamente seis DBB**: el artículo no especifica
su ubicación exacta dentro de las torres de detección. Tampoco publica el
número de grupos de mg_conv; `groups=2` sigue siendo nuestra decisión de
reproducción de la etapa 2.

## Decisión arquitectónica

- **Paper, texto:** propone sustituir el head de YOLOv8 por Detect_DBB y
  convertir sus ramas DBB en una convolución durante inferencia; no identifica
  índices de capas internas.
- **Figura 8, inferencia:** representa conceptualmente `DBB → Conv → Conv`.
  Reemplazar el primer `Conv 3×3` de cada torre por DBB se aproxima a esa
  secuencia, pero la figura no confirma seis sustituciones ni dibuja el head
  desacoplado en detalle. La Figura 9 muestra P3/P4/P5 terminando en
  Detect_DBB. Fuente principal: `C:\Users\Usuario\Downloads\full_text (11).pdf`.
- **DBB oficial:** la lógica interna y la fusión se adaptaron de
  https://github.com/DingXiaoH/DiverseBranchBlock (Apache-2.0). Su rama de
  promedio aplica `Conv 1×1 → BN/padding → AvgPool → BN`; el orden escrito en
  LMD_YOLO es `AvgPool → Conv 1×1`. Se respeta el código oficial y se declara
  la diferencia. La Figura 8 tampoco muestra claramente la operación AvgPool.
- **Nuestra decisión:** un DBB reemplaza el primer `Conv+BN+SiLU` de `cv2` y
  otro el de `cv3` en cada escala, para seis en total. La variante experimental
  B de doce DBB permanece disponible como evidencia, pero su segundo DBB por
  torre no está respaldado explícitamente por la Figura 8 ni por el texto.

El head conserva torres desacopladas, las segundas convoluciones 3×3, las
proyecciones 1×1, `reg_max=16`, DFL, `dist2bbox`, el comportamiento anchor-free
y P3/P4/P5 con strides 8/16/32. Los bloques sustituidos conservan canales,
kernel 3, stride 1, padding 1, groups 1 y SiLU después de la suma/fusión.

## Archivos y ejecución

- YAML: `models/yolov8n_leyolo_nano_mgconv_detect_dbb_stage3.yaml`.
- Head: `third_party/ultralytics/nn/modules/detect_dbb.py`. Hereda la lógica
  validada de la variante experimental A sin modificar ni borrar A/B.
- Validación previa: `scripts/validate_detect_dbb_stage3.py`; resultado en
  `outputs/stage3_detect_dbb/structural_validation.json`.
- Única época smoke: `scripts/train_detect_dbb_stage3_smoke.py`; resultados
  bajo `runs/detect/leyolo_nano_mgconv_dbb_stage3_smoke` y validación explícita
  de `best.pt` bajo `runs/detect/leyolo_nano_mgconv_dbb_stage3_best_val`.
- Conversión de `best.pt`: `scripts/convert_validate_detect_dbb_stage3.py`;
  checkpoint independiente bajo
  `runs/detect/leyolo_nano_mgconv_dbb_stage3_deploy/weights/` y detalle
  numérico en `outputs/stage3_detect_dbb/deploy_validation.json`.

Los scripts de entrenamiento y conversión rechazan sobrescribir rutas
preexistentes. La conversión carga `best.pt`, emplea FP32 y `eval()`, copia el
modelo y aplica `switch_to_deploy()` a seis DBB: en cada uno quedan una
`Conv2d 3×3` equivalente y SiLU, sin las ramas de entrenamiento. Se comparan
salidas de cada bloque, píxeles de borde, logits por escala de regresión y
clasificación, features P3/P4/P5 y salida decodificada antes de NMS; el
checkpoint desplegado se vuelve a cargar para comprobarlo. El SHA-256 del
`best.pt` original debe permanecer idéntico. Tolerancias: bloque
`rtol=1e-4, atol=1e-5`; modelo `rtol=1e-3, atol=1e-4`.

La comprobación estricta usa la ruta CPU FP32 con MKLDNN desactivado para
evitar diferencias de redondeo entre kernels CPU multirrama y fusionado. El
archivo `outputs/stage3_detect_dbb/backend_diagnostic.json` compara también
MKLDNN activado: en esa ruta un elemento de `P5_cv3` supera la tolerancia
estricta por bloque, mientras la salida del modelo completo permanece dentro
de la tolerancia fijada. No se elevaron umbrales ni se alteraron pesos.

Los parámetros/GFLOPs TRAIN miden el *forward* multirrama en `eval()`, no
backpropagation. DEPLOY mide los seis DBB fusionados, sin fusionar todas las
otras capas Conv+BN de Ultralytics. El checkpoint desplegado se guarda en
FP32 para preservar la equivalencia comprobada; `best.pt` de Ultralytics
puede estar en FP16, por lo que el tamaño en disco no tiene por qué bajar aun
cuando disminuya el número de parámetros. No se aplica INT8.

La única época pretende demostrar entrenamiento y validación funcionales;
sus métricas **no** evalúan la precisión final. Las cifras de ablación del
paper son referencia y presentan contradicciones entre tabla y prosa; no se
ajusta esta arquitectura para igualarlas.
