# Preprocesamiento canónico PC–Android para LMD_YOLO

## Alcance

Este contrato corrige exclusivamente la discrepancia de preprocesamiento. No
modifica el ONNX ni los pesos. El artefacto continúa siendo
`best_deploy_fp32_source_copy.onnx`, con SHA-256
`a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8`.

## Causa de la discrepancia anterior

El golden PC usaba `cv2.resize(..., INTER_LINEAR)` y Android usaba
`Bitmap.createScaledBitmap(..., filter=true)`. Ambos son bilineales, pero no
comparten necesariamente coordenadas, cuantización de coeficientes ni regla de
redondeo. Esto produjo hasta 9 niveles RGB de diferencia después del resize.

La imagen de prueba mide 512x512. Su letterbox a 640x640 no necesita padding:
la geometría anterior era equivalente a un resize directo. La discrepancia no
era causada por letterbox.

## Contrato canónico

1. La matriz de entrada debe estar orientada correctamente antes del
   preprocesamiento. No se permite aplicar EXIF implícitamente en un lado y no
   en el otro.
2. La entrada lógica es RGB8 opaco en sRGB.
3. Letterbox Ultralytics:
   - destino 640x640;
   - `auto=false`, `scale_fill=false`, `scaleup=true` y centrado;
   - padding RGB `(114,114,114)`;
   - relación `r=min(640/h,640/w)`;
   - tamaño sin padding redondeado al entero más cercano, empate al par;
   - padding impar: el píxel sobrante se asigna a derecha o abajo.
4. Resize bilinear canónico:
   - centros de píxel (`half-pixel`);
   - pesos racionales enteros exactos;
   - clamp al borde;
   - redondeo final half-up a RGB8.
5. Conversión RGB HWC a RGB NCHW.
6. Conversión a `float32` y normalización mediante `valor/float32(255.0)`.
7. Forma final `[1,3,640,640]`.

Las mismas ecuaciones están implementadas en
`scripts/generate_canonical_mobile_golden.py` y
`mobile_benchmark/app/src/main/java/com/lmdyolo/benchmark/CanonicalPreprocessor.kt`.

## Decodificación JPEG

La decodificación se trata como una etapa anterior y medible:

- PC: OpenCV color, orientación EXIF ignorada explícitamente, BGR a RGB8;
- Android: `BitmapFactory`, `ARGB_8888`, sRGB explícito, `inScaled=false`.

JPEG es con pérdida y diferentes implementaciones de IDCT, conversión de color,
perfiles ICC o manejo EXIF pueden producir RGB8 distintos. Por ello, la
identidad universal entre decodificadores JPEG no forma parte del contrato. La
matriz RGB8 resultante siempre se compara y se informa por separado.

Para cámaras, la orientación debe normalizarse explícitamente antes de este
contrato; no debe depender de un comportamiento implícito del decoder.

## Definiciones de equivalencia

### Equivalencia exacta de preprocesamiento

Partiendo de la misma matriz RGB8, PC y Android deben producir un tensor
idéntico bit a bit (`rtol=0`, `atol=0`). Esto valida geometría, interpolación,
padding, canales, layout y normalización.

### Equivalencia funcional desde JPEG

Cada plataforma decodifica el mismo JPEG con su decoder nativo y después aplica
el pipeline canónico. Se informan por separado la diferencia RGB8, la diferencia
del tensor, la salida ONNX pre-NMS y las detecciones. Una diferencia JPEG no se
oculta aumentando tolerancias: se evalúa por magnitud y efecto en la salida.

## Resultado en Samsung Galaxy S23

Resultado: `mobile_benchmark/results/preprocessing_validation_1790066184495.json`.

- estado `PASS_CANONICAL_EXACT`;
- RGB8 OpenCV frente a BitmapFactory: 786,432/786,432 elementos idénticos;
- tensor JPEG a tensor: 1,228,800/1,228,800 elementos idénticos;
- error de tensor máximo 0, medio 0;
- salida ONNX: 67,200 elementos, cero fuera de `rtol=1e-3`, `atol=1e-3`;
- error ONNX máximo `0.00115966796875`;
- error ONNX medio `1.6141440719366075e-05`;
- candidatos y detecciones a confianza 0.25: 0, igual que el golden;
- NNAPI no ejecutado.

En esta imagen concreta se obtuvo equivalencia exacta incluso desde JPEG. Esto
no convierte la identidad JPEG en una garantía para todas las imágenes,
dispositivos o versiones del decoder.

## Relación con OpenCV/Ultralytics

La geometría reproduce `LetterBox` de Ultralytics 8.4.153. En la imagen de
prueba, el resizer canónico frente a OpenCV produjo:

- `INTER_LINEAR`: máximo 1 nivel RGB, media 0.124546;
- `INTER_LINEAR_EXACT`: máximo 1 nivel RGB, media 0.010721.

La diferencia máxima de un nivel frente a OpenCV es explícita: se prioriza una
ecuación portable y reproducible en PC y Android, manteniendo el kernel
bilinear, los centros de píxel y la geometría del modelo.

La medición única del letterbox y construcción del tensor fue 26.416 ms en el
APK debug. No es un benchmark sostenido ni un valor release; la optimización
nativa debe evaluarse por separado si esta etapa pasa a ser un cuello de
botella.
