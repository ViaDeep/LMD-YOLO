# Auditoría local previa al paquete Stage 6

## Fuente canónica

- Ultralytics efectivo: copia local 8.4.153 en
  `C:\TP1\third_party\ultralytics`.
- YAML LMD Stage 3:
  `models/yolov8n_leyolo_nano_mgconv_detect_dbb_stage3.yaml`, SHA-256
  `2cd553b41f5e637b7e59c7ae4a3c7ab00cb6b119455790918fd3e2f61ac1516b`.
- Checkpoint TRAIN canónico de smoke: 6,849,255 bytes, SHA-256
  `6ae788c05d9bc66ef1a600e0fcd895455a8f115c97797e5b49b48a528b2050a4`.
- `yolov8n.pt` de referencia: 6,549,796 bytes, SHA-256
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.

Los checkpoints están en `reference_weights/` solo para procedencia. Ninguno
se usa para inicializar la comparación.

## Implementación personalizada requerida

| Archivo | Función | SHA-256 |
|---|---|---|
| `nn/modules/leyolo.py` | `LeYOLOConv`, `LeYOLOInvertedBottleneck` | `30ccac2f56f11857538d6c0d976cebda3c2b6d25f13312739fc2aff742df4f15` |
| `nn/modules/mg_conv.py` | `MGConv` canónico | `456aec9dca9fbd61e43ad8012cf07af9d0700c5a5801a9655f0d50a84cc7bf17` |
| `nn/modules/mg_conv_experimental.py` | base usada por MGConv | `27257163a827f41e76d4668adffc2c020cc0a536f403a64a01b4c14483fd29cb` |
| `nn/modules/detect_dbb.py` | `DetectDBB` canónico | `b479f174c94657531b4391ee457aba0a967470d7840c1b0dd6e7c22ec7b7fa66` |
| `nn/modules/detect_dbb_experimental.py` | construcción de torres DBB | `0bcf0d98ec581145f22b426c2b125fd5e9229e853c862c13ac63c8d002eeed3a` |
| `nn/modules/diverse_branch_block.py` | DBB multirrama y conversión opcional | `bd5cd2fed234f9b1050b268d6b8ba1960cc2ad37d1b9e80a941b11cfa56edae5` |
| `nn/modules/__init__.py` | exporta los módulos personalizados | `71e1d74c13fa236f29f5f912934c33755a7873fb47242cd82aec795cc427ecdc` |
| `nn/tasks.py` | imports, resolución YAML y parser de canales/head | `5980c8eb209e1955ae1e3f605cfacb02dfb299d1321cc92d7f53c166cdf1f6e2` |

`tasks.py` añade los tipos personalizados al espacio de resolución de YAML,
escala correctamente el canal de expansión LeYOLO, trata LeYOLO/MGConv como
módulos base y pasa `reg_max`, `end2end` y canales P3/P4/P5 a DetectDBB.
Copiar solamente los cuatro módulos sin estas modificaciones no reconstruye
el modelo. Por esa razón se vendorizó el paquete Ultralytics completo (sin
caches), no una instalación vanilla ni un monkey-patch parcial.

La auditoría previa ya validada localmente espera:

- LeYOLO-Nano;
- MGConv `groups=2` en capas 21 y 24;
- DetectDBB con 6 DBB TRAIN multirrama;
- 4 clases y strides 8/16/32;
- 3,073,884 parámetros y 8.56617 GFLOPs a 640.

El preflight de Colab vuelve a medir y exige exactamente estos valores. No
existe ninguna llamada a `switch_to_deploy()` en el flujo Stage 6.

La medición de FLOPs del preflight usa `ultralytics-thop==2.1.6` directamente
sobre una copia CPU FP32 del modelo y un forward real `1x3x640x640`. Registra
MACs y aplica la convención `FLOPs = 2 × MACs`. No usa el proxy por stride de
`ultralytics.utils.torch_utils.get_flops`, porque esa función devuelve 0.0 de
forma silenciosa tanto cuando falta THOP como cuando ocurre cualquier error.
En Stage 6, una dependencia ausente, una excepción, MACs no positivos o una
discrepancia de parámetros producen un error explícito.

## Dataset

Se copió exactamente `C:\TP1\data\dataset`, sin regenerar splits:

| Split | Imágenes | Labels | Uso Stage 6 |
|---|---:|---:|---|
| train | 3,082 | 3,082 | entrenamiento |
| val | 385 | 385 | seguimiento y comparación |
| test | 386 | 386 | empaquetado para integridad; **no se usa** |

La auditoría confirmó correspondencia uno-a-uno imagen/label, clases 0–3 y
coordenadas YOLO normalizadas. Objetos en todos los splits: D00=4,104,
D10=2,359, D20=934, D40=321. Digest combinado del manifest del dataset:
`8ae529f168c13e46d014c4799810becaecdb271128ccd0e40780ff5c713f07eb`.

`dataset/data.yaml` es portable. Durante ejecución se genera, dentro del
directorio de resultados, un YAML con la ruta absoluta de la copia extraída;
el YAML local original de Windows no fue modificado.

## Versiones

Entorno local auditado: Python 3.11.9, PyTorch 2.14.0+cpu, torchvision 0.29.0
y Ultralytics 8.4.153 vendorizado. El build local de PyTorch no se fuerza en
Colab porque no contiene CUDA. El preflight registra Python, torch,
torchvision, GPU y VRAM efectivos de la T4; ambos modelos se ejecutan dentro
del mismo runtime y por ello comparten exactamente PyTorch/CUDA.

## Exclusiones confirmadas

No se incluyeron `.venv`, APK, mobile benchmark, NNAPI, ONNX, checkpoints
DEPLOY, runs históricos, overfit64, outputs antiguos, temporales ni caches.
