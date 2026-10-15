# Baseline definitivo CPU/XNNPACK con preprocesamiento canónico

## Veredicto

`PASS` en Samsung Galaxy S23 SM-S911B. Se completaron 10 warmups y 100
ejecuciones FP32 medidas, desconectado de la carga y sin estado térmico adverso.

Resultado fuente:
`mobile_benchmark/results/cpu_xnnpack_1790067825791.json`.

SHA-256 del resultado:
`ec8f75953cd276c825457e36b973effd6b8b03c4b5e1ce386275cd6dbfacbc98`.

## Integridad y golden

- ONNX SHA-256:
  `a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8`.
- Pipeline: `canonical_v1_locked`.
- Entrada: `[1,3,640,640]`, FP32.
- Salida: `[1,8,8400]`.
- Tensor canónico Android frente a PC: 1,228,800 valores idénticos, error 0.
- Salida ORT frente a PC: 67,200 valores, `rtol=1e-3`, `atol=1e-3`.
- Error absoluto máximo: `0.00115966796875`.
- Error absoluto medio: `1.6141440719366075e-05`.
- Elementos fuera de tolerancia: 0.
- Candidatos/detecciones a confianza 0.25: 0/0.

## Latencias canónicas

| Etapa | Media | P50 | P90 | P95 | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| Decode JPEG | 3.454 ms | 3.484 ms | 3.596 ms | 3.608 ms | 2.945 ms | 4.128 ms |
| Letterbox canónico | 28.439 ms | 28.289 ms | 29.410 ms | 29.804 ms | 27.499 ms | 30.560 ms |
| RGB/NCHW FP32 | 4.967 ms | 4.946 ms | 5.089 ms | 5.165 ms | 4.632 ms | 6.749 ms |
| Preparación total | 36.860 ms | 36.768 ms | 37.800 ms | 38.103 ms | 35.715 ms | 39.022 ms |
| Inferencia | 76.810 ms | 76.741 ms | 77.698 ms | 78.112 ms | 75.363 ms | 79.055 ms |
| Postproceso | 0.064 ms | 0.064 ms | 0.065 ms | 0.066 ms | 0.062 ms | 0.104 ms |
| NMS | 0.032 ms | 0.031 ms | 0.034 ms | 0.036 ms | 0.022 ms | 0.051 ms |
| End-to-end | 113.772 ms | 113.634 ms | 115.022 ms | 115.635 ms | 111.734 ms | 116.696 ms |

- FPS equivalente de inferencia: 13.019.
- FPS equivalente end-to-end: 8.790.
- Desviación estándar de inferencia: 0.786 ms; CV 1.02 %.
- Desviación estándar end-to-end: 0.950 ms; CV 0.84 %.
- Carga de sesión: 59.060 ms.

`preprocessing_ms` incluye decode, letterbox y tensor. El núcleo canónico sin
decode cuesta 33.406 ms de media.

## Estabilidad, energía y térmica

- Inferencia primeras 20: 76.399 ms.
- Inferencia últimas 20: 77.251 ms.
- Degradación de inferencia: +1.116 %.
- End-to-end primeras 20: 113.606 ms.
- End-to-end últimas 20: 113.843 ms.
- Degradación end-to-end: +0.208 %.
- No se detectó desaceleración sostenida superior al 15 %.
- Estado térmico: `NONE` antes, después de warmup, cada 10 runs y al terminar.
- Batería: 100 % antes y después; sin carga.
- Temperatura de batería: 28.7 °C antes y después; no es temperatura del SoC.

## Memoria

- PSS antes de sesión: 117,183 KiB (114.4 MiB).
- PSS después de sesión: 155,484 KiB (151.8 MiB).
- PSS durante runs: 218,103–261,587 KiB (213.0–255.5 MiB).
- PSS al terminar benchmark antes del perfil: 256,679 KiB (250.7 MiB).
- Heap nativo run 10 a run 100: +1,322,672 bytes.

La oscilación de PSS/heap Java concuerda con asignaciones y GC. El crecimiento
nativo de aproximadamente 1.26 MiB entre runs 10 y 100 no muestra una fuga
acelerada en esta ventana, aunque 100 ejecuciones no constituyen una prueba de
soak prolongada.

## Proveedores observados

El perfil de una inferencia separada posterior al benchmark registró:

- `XnnpackExecutionProvider`: 81 eventos/nodos perfilados; tipos `Conv`,
  `MaxPool`.
- `CPUExecutionProvider`: 112 eventos/nodos perfilados; tipos `Add`, `Concat`,
  `Div`, `Gelu`, `Mul`, `QuickGelu`, `Reshape`, `Resize`, `Sigmoid`, `Softmax`,
  `Split`, `Sub`, `Transpose`.
- Duración perfilada: 50,768 us XNNPACK y 32,159 us CPU.

Los conteos y duraciones pertenecen a la inferencia con profiler y no son
porcentajes del grafo, del trabajo ni de la latencia del benchmark. Confirman
que CPU/XNNPACK es una ejecución mixta y no XNNPACK exclusivo.

ONNX Runtime advirtió que algunos nodos no fueron asignados al proveedor
preferido y señaló que puede asignar operaciones relacionadas con formas a CPU.
No hubo errores de ejecución.

## Comparación A/B/C

| Métrica | A: CPU antiguo | B: CPU canónico | C: NNAPI |
|---|---:|---:|---:|
| Estado | PASS runtime; JPEG no exacto | **PASS definitivo** | ABORTED al crear sesión |
| Golden tensor | PASS aislado | PASS exacto desde JPEG | Pipeline exacto; sin inferencia |
| Session load | 130.597 ms | 59.060 ms | No disponible |
| Decode | 1.905 ms | 3.454 ms | No benchmark |
| Preparación total | 7.237 ms | 36.860 ms | No benchmark |
| Inferencia media | 76.630 ms | 76.810 ms | No disponible |
| Inferencia p50 | 76.584 ms | 76.741 ms | No disponible |
| Inferencia p90 | 77.285 ms | 77.698 ms | No disponible |
| Inferencia p95 | 77.372 ms | 78.112 ms | No disponible |
| Inferencia mín./máx. | 75.359/79.556 ms | 75.363/79.055 ms | No disponible |
| FPS inferencia | 13.050 | 13.019 | No disponible |
| Postproceso/NMS | 0.064/0.032 ms | 0.064/0.032 ms | No disponible |
| End-to-end medio | 84.112 ms | 113.772 ms | No disponible |
| End-to-end p50 | 83.996 ms | 113.634 ms | No disponible |
| End-to-end p90 | 84.911 ms | 115.022 ms | No disponible |
| End-to-end p95 | 85.203 ms | 115.635 ms | No disponible |
| FPS end-to-end | 11.889 | 8.790 | No disponible |
| Degradación inferencia 20/20 | +0.172 % | +1.116 % | No disponible |
| PSS durante runs | 239,779–273,482 KiB | 218,103–261,587 KiB | No disponible |
| Estado térmico | NONE | NONE | No benchmark |
| Proveedores ejecutados | XNNPACK + CPU | XNNPACK + CPU | Ninguno; sesión falló |
| Warning/error | Asignación parcial a CPU | Asignación parcial a CPU | `AddNnapiSplit count [0]` |

## Comparabilidad

Directamente comparables, con la cautela de ser ejecuciones separadas:

- latencia de inferencia, forma de salida, precisión, runtime, dispositivo,
  proveedor, postproceso/NMS y estabilidad;
- la inferencia cambió solo +0.236 %, magnitud compatible con variación entre
  corridas y con el pequeño cambio de valores de entrada.

No directamente comparables:

- preprocesamiento y end-to-end, porque A usa Bitmap/Skia y B usa el pipeline
  canónico determinista. La diferencia end-to-end de +29.659 ms (+35.26 %) mide
  principalmente el coste de esa implementación, no un cambio del modelo;
- carga de sesión, porque es una sola medición cold-start por corrida y no
  depende del pipeline;
- memoria absoluta, porque cambiaron las asignaciones del preprocesamiento y el
  APK/harness entre A y B.

El intento NNAPI no es comparable en rendimiento: nunca completó la creación de
sesión ni ejecutó un nodo.
