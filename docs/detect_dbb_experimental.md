# Detect_DBB: comparación experimental A/B (sin entrenamiento)

Estas dos arquitecturas son **hipótesis de nuestra reproducción**, no una
identificación del Detect_DBB original. El YAML de etapa 2, su módulo mg_conv,
sus pesos y resultados permanecen intactos.

## Fuentes y límites de certeza

- **LMD_YOLO, texto (A):** describe DBB multirrama con convolución K×K,
  secuencia 1×1→K×K y rama de promedio; las ramas se suman y se fusionan en
  una convolución para inferencia. Afirma que sustituye el head original.
- **Figuras 8/9 (B):** la Figura 8 sitúa DBB antes de dos `Conv2d`, pero no
  dibuja las torres desacopladas ni especifica las convoluciones sustituidas.
  Sus cuatro rutas dibujadas tampoco identifican una operación AvgPool.
  La Figura 9 sitúa Detect_DBB en P3/P4/P5, sin detallar su interior.
- **DBB oficial (C):** `DiverseBranchBlock` implementa, en el caso denso,
  K×K+BN, 1×1+BN, 1×1+BN/padding→K×K+BN, y
  1×1+BN/padding→AvgPool+BN. En este último caso **Conv precede AvgPool**,
  a diferencia de la ecuación de LMD_YOLO. La fusión sigue las transformaciones
  oficiales (`get_equivalent_kernel_bias`, `switch_to_deploy`). Fuente:
  https://github.com/DingXiaoH/DiverseBranchBlock (Apache-2.0).
- **Nuestra decisión (D):** ensayar seis o doce sustituciones densas en las
  torres existentes de Ultralytics 8.4.153, manteniendo SiLU después de cada
  DBB, no dentro de la suma fusionable. No se añade DBB compartido.

## Arquitecturas

Ambas mantienen LeYOLO-Nano, mg_conv en 21/24, `reg_max=16`, DFL, cuatro
clases, P3/P4/P5, las proyecciones finales 1×1 y la decodificación original.
Se modificó únicamente la clase del head en el YAML experimental.

| Variante | cv2 y cv3 en cada escala | DBB totales |
| --- | --- | ---: |
| A | DBB→Conv 3×3→Conv2d 1×1 | 6 |
| B | DBB→DBB→Conv2d 1×1 | 12 |

Los DBB reemplazan `Conv+BN+SiLU` locales de kernel 3, stride 1, padding 1,
groups 1; toman sus canales de entrada/salida. Esta especialización del módulo
oficial no admite otras configuraciones ni pretende ser una biblioteca DBB
genérica. El módulo mantiene la rama identidad-1×1 y el padding con el valor
BN correspondiente, necesarios para reproducir fielmente la fusión en bordes.

## TRAIN→DEPLOY y comprobación

`switch_to_deploy()` elimina las cuatro ramas de entrenamiento de cada DBB y
deja una única Conv2d 3×3 con sesgo, seguida de SiLU. El script convierte una
**copia profunda** en `eval()`; preserva el modelo multirrama. No crea pesos
entrenados ni ejecuta épocas.

Ejecutar desde `C:\TP1`:

```powershell
& '.\.venv\Scripts\python.exe' scripts/compare_detect_dbb_experimental.py
```

El script comprueba forward de entrenamiento y evaluación 1×3×640×640,
cuatro clases, logits de cajas/clases, P3/P4/P5, strides 8/16/32 y desaparición
de las ramas tras la conversión. Usa FP32 y compara bloques aislados, incluidos
bordes y estadísticas BN no triviales (`rtol=1e-4`, `atol=1e-5`); después
compara logits por escala, logits concatenados y salida decodificada antes de
NMS (`rtol=1e-3`, `atol=1e-4`). Registra máximos absolutos y relativos en
`outputs/detect_dbb_experimental/structural_comparison.json`.

Los GFLOPs provienen del mismo `get_flops`/THOP local usado para etapa 2,
con entrada 640; TRAIN significa estructura multirrama medida en `eval()`
(necesario para la comparación determinista), no FLOPs de backpropagation.
DEPLOY mide DBB convertidos, pero **no** aplica la fusión general de los
restantes `Conv+BN` de Ultralytics. Por ello las pequeñas diferencias entre
A-DEPLOY y B-DEPLOY no demuestran por sí solas distinta latencia móvil; una
comparación final de dispositivo debe fusionar el resto de manera uniforme y
medir FPS/energía en el mismo hardware.

La tabla 5 del paper comunica 1.34 M / 3.8 GFLOPs para LeYOLO+mg_conv y
1.70 M / 3.8 GFLOPs al añadir Detect_DBB, pero no aclara si los parámetros
corresponden a TRAIN o DEPLOY y contradice otras cifras de su propia prosa.
Estas cifras se usan como referencia, nunca como objetivo de ajuste.
