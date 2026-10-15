# LMD_YOLO Android on-device benchmark

Harness Android nativo y aislado para validar el ONNX FP32 congelado en un
Samsung Galaxy S23. No pertenece a la futura aplicación React Native/Expo.

## Alcance congelado

- Modelo: `app/src/main/assets/best_deploy_fp32_source_copy.onnx`
- SHA-256: `a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8`
- Entrada: `[1,3,640,640]`, letterbox canónico centrado, RGB, padding 114,
  bilinear determinista, NCHW FP32 `/255`
- Salida pre-NMS: `[1,8,8400]`
- Clases: `D00`, `D10`, `D20`, `D40`
- NMS externo class-aware: confianza 0.25, IoU 0.45
- ONNX Runtime Android fijado: `1.24.3`
- Fase 1: XNNPACK explícito, 4 threads, ORT intra-op 1, FP32
- Fase 2: NNAPI con `CPU_DISABLED`, sin `USE_FP16`; ORT CPU permanece como fallback

No se modifica ni reexporta el modelo. El APK vuelve a calcular el SHA-256 del
asset en el teléfono y aborta si no coincide.

## Golden test

Los assets incluyen:

- el JPEG original `China_Drone_000001.jpg`;
- `golden_input_fp32.bin`, generado con OpenCV en PC;
- `golden_output_fp32.bin`, generado con ONNX Runtime CPU en PC;
- `golden_metadata.json`, con hashes, valores representativos y detecciones.

El golden actual tiene cero detecciones a confianza 0.25, esperable para el
checkpoint de una época. La validación real usa los 67,200 valores de salida, no
solo el conteo de cajas.

El teléfono ejecuta dos comprobaciones:

1. `golden_input -> ORT Android -> comparación completa con golden_output`.
   Esto aísla el runtime y detiene el benchmark si falla la tolerancia prefijada
   `rtol=1e-3, atol=1e-3`.
2. `JPEG -> Bitmap Android sRGB -> letterbox bilinear canónico -> tensor -> ORT`, registrando por
   separado diferencias del tensor y de la salida. No se cambia ninguna
   tolerancia automáticamente.

## Qué mide

- Copia/hash del asset y carga de sesión.
- Decodificación JPEG.
- Resize.
- Normalización y construcción NCHW.
- Inferencia ORT.
- Decodificación de candidatos.
- NMS.
- Tiempo end-to-end.
- 10 warmups y 100 ejecuciones individuales.
- Media, mediana, p50, p90, p95, mínimo, máximo y FPS equivalente.
- PSS aproximado, private dirty, heap Java y heap nativo.
- Estado térmico categórico de `PowerManager` y evolución cada diez runs.
- Temperatura de batería, etiquetada explícitamente como batería, nunca como SoC.
- Perfil ORT no medido con recuento de eventos por execution provider.

El benchmark se niega a comenzar si el teléfono está cargando y aborta si el
estado térmico llega a `SEVERE`. NNAPI queda bloqueado hasta que CPU/XNNPACK haya
producido un `PASS` de 100 ejecuciones.

## Requisitos en este PC

Ya existe JDK 21. Falta instalar Android Studio/SDK/ADB.

### Opción recomendada: Android Studio

1. Instale Android Studio estable para Windows desde
   <https://developer.android.com/studio/>.
2. En el asistente acepte la licencia del Android SDK.
3. Abra **Tools > SDK Manager**.
4. En **SDK Platforms**, instale **Android 15 / API 35**.
5. En **SDK Tools**, instale:
   - Android SDK Build-Tools 35.0.0;
   - Android SDK Platform-Tools;
   - Android SDK Command-line Tools (latest).
6. No se necesita emulador ni NDK para este harness.
7. El SDK normalmente quedará en `%LOCALAPPDATA%\Android\Sdk`.

También puede usarse el Android CLI oficial, pero debe aceptar personalmente la
licencia y después instalar `platforms;android-35`, `build-tools;35.0.0` y
`platform-tools`.

## Preparar el Galaxy S23

1. En el teléfono: **Ajustes > Acerca del teléfono > Información de software**.
2. Pulse **Número de compilación** siete veces e introduzca el PIN.
3. Vuelva a **Ajustes > Opciones de desarrollador**.
4. Active **Depuración USB**.
5. Conecte un cable USB de datos y seleccione transferencia de archivos si
   Windows no detecta la interfaz ADB.
6. Acepte en el teléfono la huella RSA de este PC. No marque autorización
   permanente si no lo desea.
7. Si Windows no instala el controlador, use Windows Update o el Samsung Android
   USB Driver oficial. No desbloquee el bootloader.

Comprobación, desde PowerShell:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
```

Debe aparecer exactamente un dispositivo con estado `device`, no `unauthorized`.

## Compilar

```powershell
cd C:\TP1\mobile_benchmark
powershell -ExecutionPolicy Bypass -File .\scripts\build_debug.ps1
```

APK esperado:

`C:\TP1\mobile_benchmark\app\build\outputs\apk\debug\app-debug.apk`

La primera compilación descarga Gradle, Android Gradle Plugin, Kotlin y el AAR
oficial `onnxruntime-android:1.24.3` desde repositorios oficiales.

## Instalar y ejecutar CPU/XNNPACK

Instale por USB, pero no ejecute la prueba sostenida mientras el USB esté
cargando el teléfono:

```powershell
cd C:\TP1\mobile_benchmark
powershell -ExecutionPolicy Bypass -File .\scripts\install_and_run.ps1 -Backend none
```

Después use una de estas dos opciones:

- Desconecte el USB y pulse **Ejecutar CPU/XNNPACK** en el teléfono.
- O habilite **Depuración inalámbrica**, empareje ADB por Wi-Fi, desconecte el
  cable y ejecute:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_and_run.ps1 -Backend xnnpack
```

Para observar logs por ADB inalámbrico:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb logcat -v threadtime -s LMD_BENCH:I
```

No use el teléfono durante la prueba. Mantenga la pantalla encendida, cierre
apps pesadas y empiece con el dispositivo a temperatura ambiente.

## Recuperar resultados

Reconecte ADB después de finalizar y ejecute:

```powershell
cd C:\TP1\mobile_benchmark
powershell -ExecutionPolicy Bypass -File .\scripts\pull_results.ps1
```

Los JSON quedarán en `C:\TP1\mobile_benchmark\results\`. El perfil ORT también
permanece en el almacenamiento interno privado de la app; su ruta se registra en
el JSON. Para listar archivos manualmente:

```powershell
& $adb shell run-as com.lmdyolo.benchmark ls -l files/results
```

## Fase NNAPI — solo después del PASS CPU

No ejecute este paso hasta revisar el JSON CPU y confirmar:

- `status: PASS`;
- `measured_runs_completed: 100`;
- golden runtime `passed: true`;
- ausencia de aborto térmico o errores.

Entonces, con el mismo teléfono sin cargar:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_and_run.ps1 -Backend nnapi
```

NNAPI usa exactamente el mismo ONNX, JPEG, golden, thresholds, 10 warmups y 100
runs. El perfil ORT registra eventos atribuidos a NNAPI y CPU; es un diagnóstico
de fallbacks, no una garantía de que esos conteos sean nodos únicos.

Después de recuperar ambos JSON:

```powershell
C:\TP1\.venv\Scripts\python.exe .\scripts\summarize_results.py
```

Esto genera `results\comparison_latest.md` con CPU/XNNPACK frente a NNAPI.

## Interpretación térmica y memoria

- `PowerManager.currentThermalStatus` es una categoría del sistema, no una
  temperatura exacta del SoC.
- `battery_temperature_c` es únicamente el sensor de batería.
- PSS es una aproximación del consumo atribuible al proceso, no el pico total del
  sistema ni la memoria reservada por drivers externos.
- Para validar throttling deben observarse simultáneamente la evolución térmica y
  el desplazamiento de latencias por orden de ejecución.
