param(
    [ValidateSet("xnnpack", "nnapi", "nnapi_validate", "preprocess", "none")]
    [string]$Backend = "xnnpack",
    [string]$SdkRoot = "$env:LOCALAPPDATA\Android\Sdk"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $SdkRoot "platform-tools\adb.exe"
$Apk = Join-Path $ProjectRoot "app\build\outputs\apk\debug\app-debug.apk"

if (-not (Test-Path -LiteralPath $Adb)) { throw "adb no encontrado: $Adb" }
if (-not (Test-Path -LiteralPath $Apk)) { throw "APK no encontrado. Ejecute scripts\build_debug.ps1 primero." }

$Devices = @(& $Adb devices | Select-String "\tdevice$")
if ($Devices.Count -ne 1) {
    & $Adb devices -l
    throw "Debe haber exactamente un dispositivo ADB autorizado; encontrados: $($Devices.Count)"
}

& $Adb install -r $Apk
if ($LASTEXITCODE -ne 0) { throw "adb install falló" }
& $Adb shell am force-stop com.lmdyolo.benchmark
& $Adb shell am start -W -n com.lmdyolo.benchmark/.MainActivity --es backend $Backend
if ($LASTEXITCODE -ne 0) { throw "No se pudo iniciar el benchmark" }

Write-Host "Logs: & '$Adb' logcat -v threadtime -s LMD_BENCH:I"
Write-Host "Si usa USB, desconéctelo antes de pulsar el benchmark sostenido; el harness se bloquea mientras carga."
