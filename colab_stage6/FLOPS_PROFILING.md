# Corrección portable del profiling de FLOPs

## Causa raíz

La primera versión de `scripts/stage6.py` importaba
`ultralytics.utils.torch_utils.get_flops`. En Ultralytics 8.4.153 esa función:

1. intenta importar `thop`;
2. devuelve `0.0` si el import falla;
3. envuelve todo el profiling en `except Exception` y también devuelve `0.0`
   ante cualquier error.

El entorno local contenía `ultralytics-thop 2.1.6` como dependencia indirecta
de la instalación anterior. `requirements-colab.txt` no lo declaraba y Colab
no tenía `thop`; por eso no llegó a ejecutar ningún forward de profiling. El
valor 0.0 no se originó en LeYOLO, MGConv, DetectDBB, DBB, CUDA ni PyTorch
2.11.0: fue el fallback silencioso por dependencia ausente.

## Solución

- Se fija `ultralytics-thop==2.1.6` explícitamente.
- `measure_gflops_cpu` crea una copia del modelo en CPU FP32/eval.
- Perfila un input real `1x3x640x640`, sin proxy de stride.
- THOP recorre las operaciones estándar internas de LeYOLO, MGConv y las
  ramas de los seis DBB TRAIN. No se reparametriza ningún bloque.
- Se registran MACs, parámetros, versión y método.
- Se comprueba que los parámetros contados por THOP coincidan con PyTorch.
- Dependencias ausentes, excepciones o MACs inválidos ahora fallan con causa
  explícita; nunca se convierten en 0.0.

Validación local con el paquete:

| Modelo | MACs @640 | GFLOPs (`2×MACs`) | Parámetros |
|---|---:|---:|---:|
| YOLOv8n | 4,097,459,200 | 8.1949184 | 3,011,628 |
| LMD_YOLO TRAIN | 4,283,084,800 | 8.5661696 | 3,073,884 |

La cifra LMD redondeada a seis decimales continúa siendo 8.566170 sin usar
una constante como resultado ni alterar arquitectura o pesos.
