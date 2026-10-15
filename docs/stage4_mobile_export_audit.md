# Auditoría de exportación y despliegue on-device (etapa 4)

Fecha: 2026-09-21. Alcance: solo el checkpoint DEPLOY FP32; no se entrenó,
cuantizó, integró una app ni convirtió a otro formato aparte de una prueba ONNX.

## Dictamen

La ruta de menor riesgo es:

`PyTorch DEPLOY -> ONNX FP32 estático 1x3x640x640, sin NMS -> ONNX Runtime Mobile -> puente React Native oficial`

Es la primera ruta porque ya quedó comprobada numéricamente, sirve en Android e
iOS, conserva una salida pre-NMS común y es la única de las opciones estudiadas
que ofrece un paquete React Native oficial con config plugin para Expo
managed/prebuild. Core ML debe evaluarse después como optimización específica de
iOS; LiteRT, como alternativa específica de Android; ExecuTorch todavía requiere
una prueba de captura y más trabajo nativo.

## 1. Estado del modelo DEPLOY

- Archivo: `runs/detect/leyolo_nano_mgconv_dbb_stage3_deploy/weights/best_deploy_fp32.pt`.
- SHA-256: `07ac5b5ac8e1691739f69d32e8d2e8a17fa6761d2029b7febe9f2427edafc852`.
- Tamaño: 9,453,763 bytes.
- Ultralytics local: 8.4.153; PyTorch: 2.14.0+cpu.
- FP32; 2,265,820 parámetros; 6.157107 GFLOPs @640.
- Entrada auditada: `[1,3,640,640]`; salida decodificada pre-NMS: `[1,8,8400]`
  (`xywh + 4 scores`, 8400 candidatos).
- Cabeza Python: `DetectDBB`; seis `DiverseBranchBlock`.
- Los seis DBB tienen solamente `dbb_reparam: Conv2d(3x3,s1,p1,g1,bias=True)`
  seguido de SiLU. No existe ninguna rama `dbb_origin`, `dbb_1x1`, `dbb_avg` ni
  `dbb_1x1_kxk`.
- La exportación de Ultralytics fusiona en su copia en memoria las capas que
  reconoce; no mutó el `.pt` original.

## 2. Módulos personalizados restantes y su reducción

| Clase Python | Cantidad | Cómputo efectivo al exportar | Riesgo de operador custom |
|---|---:|---|---|
| `LeYOLOConv` | 27 | Conv, BN y activación | Bajo |
| `LeYOLOInvertedBottleneck` | 12 | pointwise, depthwise/grouped Conv, BN, Add opcional | Bajo |
| `MGConv` | 2 | Conv 3x3 `groups=2` + BN + GELU, Conv 1x1 de atajo, Add | Bajo-medio por GELU/delegados |
| `DetectDBB` | 1 | cabeza Detect heredada y decodificación YOLO | Bajo para ONNX; debe validarse por runtime |
| `DiverseBranchBlock` | 6 | Conv 3x3 ya reparametrizada + SiLU | Bajo |

En el ONNX no quedó ningún dominio ni operador personalizado. Los dos `MGConv`
aparecen como convoluciones ONNX estándar con `group=2`; GELU se descompuso en
`Erf`, `Add`, `Mul` y constantes. Las depthwise de LeYOLO permanecen como `Conv`
con `group` 16/96/192/576. SiLU se descompuso en `Sigmoid * x`. No quedó
`BatchNormalization` en el grafo final.

La principal cautela futura no son las clases Python, sino el soporte de
`Erf`, convoluciones agrupadas/depthwise, `Resize`, `Gather`, `Split` y la
decodificación en cada delegado de hardware.

## 3. Comparación de rutas móviles

Todos los estados distintos de ONNX son conclusiones de auditoría estática, no
una afirmación de que ya hayan sido exportados.

| Ruta (formato -> runtime) | Compatibilidad con este DEPLOY | Android / iOS y hardware | FP16 / INT8 posterior | React Native, Expo y código nativo | Madurez y medición |
|---|---|---|---|---|---|
| ONNX -> ONNX Runtime Mobile | **Verificada en FP32/CPU**. LeYOLO, MGConv y DBB se reducen a ONNX estándar; opset 18, sin custom ops. | Ambos. CPU/XNNPACK; Android puede ensayar NNAPI y EPs específicos; iOS puede ensayar CoreML EP. La cobertura/particionado debe comprobarse por teléfono. | FP16 posible pero dependiente del EP y normalmente poco útil en CPU; INT8 estático posible después, no realizado. | Mejor ruta: `onnxruntime-react-native` oficial, autolinking y config plugin para Expo prebuild. Expo Go no sirve; requiere development build. Código nativo propio no es obligatorio para el primer prototipo, pero conviene para cámara/pre/postprocesado de alto FPS. | Runtime muy maduro y mantenido. Latencia fácil de instrumentar y perfilar. |
| `.tflite` -> LiteRT | **Probable, no verificada**. En 8.4.153 `tflite` es alias obsoleto de `litert`; la conversión es directa desde PyTorch con `litert_torch`, no desde el ONNX probado. El host de exportación permitido es Linux x86 o macOS, no este Windows. Conv/depthwise/grouped son naturales; hay que probar GELU/Erf y el decode. | Android e iOS. CPU/XNNPACK y GPU; NPU por delegados de fabricante, con cobertura variable. Android es su caso más natural. | En 8.4.153 FP16 no es otro archivo: FP32 puede ejecutarse FP16 mediante GPU/XNNPACK. INT8 estático/dinámico está soportado por el exportador, no realizado. | No hay una ruta oficial de RN tan directa como ORT; normalmente bridge/módulo Kotlin/Swift o paquete comunitario. No Expo Go; development build y config plugin/módulo nativo. | Maduro, especialmente Android. `benchmark_model` facilita medir, pero particionado de delegados y copias CPU/GPU exigen inspección. |
| `.pte` -> ExecuTorch | **Aplicable pero no verificada**. 8.4.153 usa `torch.export.export`; las clases deberían inlinearse a ATen estándar. La ruta directa de Ultralytics baja solo a XNNPACK, por lo que no demuestra GPU/NPU. | Ambos. El ecosistema dispone de XNNPACK CPU, Vulkan Android, Core ML/MPS iOS y backends Qualcomm/MediaTek/etc.; normalmente se genera un `.pte` específico por backend. | El export directo de Ultralytics no ofrece conversión de precisión. FP16/INT8 requieren flujos de backend/cuantiación separados. | No existe puente RN oficial equivalente al de ORT: requiere módulo nativo C++/Kotlin/Swift. Expo Go imposible; development build. | Proyecto oficial activo y ya utilizable, pero con mayor variación de backend y menor simplicidad que ORT. Tiene tooling de profiling; dificultad media-alta. |
| `.mlpackage` -> Core ML | **Alta probabilidad, no verificada**. Ultralytics traza TorchScript y convierte con coremltools. No se puede exportar desde Windows con esta versión; hacerlo en macOS/Linux y probar/cargar en macOS/iOS. | Solo Apple. Core ML decide CPU/GPU/Neural Engine; suele ser la mejor ruta iOS tras establecer baseline. | FP16 soportado. El “INT8” del exportador 8.4.153 es cuantización/paletización de pesos, no garantía W8A8 en ANE. | Requiere Swift/Obj-C o bridge RN (salvo adoptar un wrapper externo). No Expo Go; development build. | Muy maduro en iOS; Xcode Performance Report e Instruments hacen sencilla la medición. |
| NCNN -> runtime NCNN | **Probable, no verificada**. Ultralytics usa PNNX desde PyTorch. Operaciones CNN estándar favorables. | Android/iOS; CPU ARM optimizada y Vulkan GPU. Sin ruta NPU general. | FP16 sí; el exportador directo de 8.4.153 no ofrece INT8. | Integración C++/JNI/Obj-C++ y bridge RN propio; no Expo Go. | Maduro y ligero para CNN móvil. Medición nativa razonable, integración más costosa que ORT. |
| MNN -> runtime MNN | **Probable, no verificada**. Ultralytics convierte mediante ONNX, por lo que el ONNX exitoso reduce riesgo, pero aún hay otra conversión. | Android/iOS; CPU y GPU (Metal/OpenCL/Vulkan); NPU parcial/específico. | FP16 e INT8 de pesos soportados por la ruta Ultralytics; no realizados. | Bridge nativo propio o comunitario; no Expo Go. | Maduro y activo, pero añade conversor/runtime y QA adicional. Dificultad media. |

Rutas no prioritarias:

- TorchScript existe, pero PyTorch Mobile ya no se mantiene activamente; la ruta
  oficial sucesora es ExecuTorch.
- QNN podría ser valioso en un Android Qualcomm concreto, pero la exportación HTP
  de 8.4.153 es una ruta cuantizada y específica de plataforma; queda fuera de
  esta fase FP32.
- Edge TPU, RKNN, IMX, Hailo, Axelera y DEEPX son hardware específico y varias
  rutas obligan cuantización; no son una base general para smartphone.
- OpenVINO/TensorRT no son las rutas generales adecuadas para Android+iOS.

## 4. Resultado de la prueba ONNX

- Estado: **PASS**.
- Configuración: FP32, batch 1, 640x640 fijo, `dynamic=False`, `simplify=False`,
  `nms=False`, sin cuantización, CPU.
- Archivo: `outputs/stage4_mobile_export_audit/onnx_fp32_640/best_deploy_fp32_source_copy.onnx`.
- Tamaño: 9,214,905 bytes (8.79 MiB).
- ONNX 1.23.0, ONNX Runtime 1.30.0, opset 18, IR 8.
- Checker ONNX: PASS; sesión ORT CPU: PASS; warnings ORT: ninguno.
- Entrada: `images [1,3,640,640]`; salida: `output0 [1,8,8400]`.
- 284 nodos, todos del dominio `ai.onnx`.
- Operadores: `Conv 78`, `Mul 62`, `Sigmoid 56`, `Constant 29`, `Add 15`,
  `Concat 13`, `Reshape 8`, `Div 4`, `Split 4`, `MaxPool 3`, `Erf 2`,
  `Resize 2`, `Slice 2`, y uno de `Gather`, `Shape`, `Softmax`, `Sub`,
  `Transpose`.
- Convoluciones: 41 Conv 1x1/s1 densas; 2 Conv 1x1/s2 densas; 20 Conv
  3x3/s1 densas; 1 Conv 3x3/s2 densa; 2 Conv 3x3/s2 con group=2 (MGConv);
  12 depthwise/grouped de LeYOLO con grupos 16/96/192/576.
- Imagen: `China_Drone_000001.jpg`; mismo tensor para ambas rutas (BGR->RGB,
  resize directo a 640, NCHW FP32/255).
- Comparación pre-NMS decodificada: forma idéntica; max abs
  `0.001190185546875`, media abs `1.7510343607606652e-05`, max rel con
  denominador mínimo 1e-8 `0.004183507031092451`, media rel
  `9.703554710306208e-05`. Pasa `np.allclose` tanto a rtol=1e-4/atol=1e-5
  como a rtol=1e-3/atol=1e-4. El máximo relativo aislado ocurre cerca de cero
  y no contradice el allclose.
- El SHA-256 del `.pt` original fue idéntico antes y después.
- Warning de Ultralytics: la cabeza no tiene rama one-to-one y usa la salida
  one-to-many. Es esperado para esta cabeza YOLOv8 legacy y no es un error.
- Mensaje informativo adicional: recomendación de OpenVINO para el CPU Intel del
  PC; no afecta el artefacto ni la decisión móvil.

## 5. Implicaciones por plataforma

### Android

Primero usar ONNX Runtime con CPU/XNNPACK como baseline reproducible. Después
comparar NNAPI en el mismo teléfono, verificando cuántos nodos delega: una
partición pequeña puede ser más lenta por transferencias. LiteRT es el segundo
candidato Android si ORT no satisface latencia/energía. NCNN puede ser útil para
CPU/Vulkan con una integración C++ más especializada. NPU requiere decidir antes
un SoC/mercado objetivo; no debe asumirse portátil.

### iOS

ONNX Runtime permite conservar el mismo artefacto y postproceso que Android.
Luego debe compararse con Core ML nativo, exportado y probado desde macOS, porque
Core ML puede aprovechar CPU/GPU/ANE y ofrece tooling de profiling excelente. Un
éxito de CoreML EP dentro de ORT no equivale automáticamente a la misma cobertura
o rendimiento de un `.mlpackage` nativo.

### React Native y Expo

El ONNX es un formato; `onnxruntime-react-native` es el runtime/puente. Expo Go
no incluye ese código nativo. El camino viable es Expo prebuild/development build
con el config plugin del paquete, o bare React Native. Core ML, LiteRT, NCNN, MNN
y ExecuTorch requieren development build y, en la práctica, un módulo nativo si
no se adopta un wrapper mantenido.

Para video, no conviene cruzar el bridge JS con cada frame tensorizado (un input
FP32 640x640 ocupa ~4.69 MiB) ni con todo `8x8400` (~262.5 KiB) cada frame. La
cámara, resize/letterbox, inferencia y NMS deberían permanecer en nativo/JSI; a
JavaScript solo deberían volver las cajas finales. La prueba actual usó resize
directo para igualar la validación previa; la app debe fijar explícitamente si el
contrato de producción será letterbox Ultralytics y deshacer sus offsets/escala.

## 6. Siguiente prueba física recomendada

1. Congelar el ONNX probado y su SHA-256 junto con una imagen/tensor golden y la
   salida PyTorch/ORT golden.
2. Crear, todavía sin UI final, un harness Android mínimo que empaquete el ONNX,
   use exactamente el mismo preprocesado, ejecute ORT CPU/XNNPACK y compare la
   salida pre-NMS y detecciones post-NMS con el golden.
3. Medir por separado carga fría, preproceso, inferencia, NMS y extremo a extremo:
   calentamiento, al menos 100 iteraciones, percentiles p50/p90/p95, FPS sostenido,
   memoria y temperatura/thermal throttling.
4. En ese mismo teléfono comparar XNNPACK con NNAPI, registrando particiones y
   fallbacks; no elegir el acelerador por nombre.
5. Repetir en un Android de gama media y uno objetivo de producción.
6. En un Mac/iPhone, repetir primero con ORT y luego exportar/evaluar Core ML FP32;
   solo después decidir si mantener un artefacto común o uno por plataforma.
7. Cuando el runtime esté elegido, recién construir el development build Expo/RN
   y conservar pre/inferencia/NMS en la capa nativa.

## Evidencia y fuentes

- Resultados reproducibles: `outputs/stage4_mobile_export_audit/deploy_structure.json`
  y `outputs/stage4_mobile_export_audit/onnx_fp32_640_validation.json`.
- Script: `scripts/audit_mobile_export_stage4.py`.
- Ultralytics local 8.4.153: `third_party/ultralytics/engine/exporter.py` y
  `third_party/ultralytics/utils/export/`.
- Ultralytics export: https://docs.ultralytics.com/modes/export/
- ONNX Runtime Mobile: https://onnxruntime.ai/docs/tutorials/mobile/
- ONNX Runtime React Native: https://github.com/microsoft/onnxruntime/tree/main/js/react_native
- ExecuTorch backends: https://docs.pytorch.org/executorch/stable/backends-section.html
- Core ML: https://developer.apple.com/documentation/coreml/
- Expo native code: https://docs.expo.dev/workflow/customizing/
- NCNN: https://github.com/Tencent/ncnn
- MNN: https://github.com/alibaba/MNN
