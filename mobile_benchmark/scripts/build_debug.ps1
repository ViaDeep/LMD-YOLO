param(
    [string]$SdkRoot = "$env:LOCALAPPDATA\Android\Sdk"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SdkManager = Join-Path $SdkRoot "cmdline-tools\latest\bin\sdkmanager.bat"
$Adb = Join-Path $SdkRoot "platform-tools\adb.exe"

if (-not (Test-Path -LiteralPath $SdkRoot)) {
    throw "Android SDK no encontrado en: $SdkRoot"
}
if (-not (Test-Path -LiteralPath $Adb)) {
    throw "Falta platform-tools/adb. Instale Android SDK Platform-Tools desde SDK Manager."
}
if (-not (Test-Path -LiteralPath (Join-Path $SdkRoot "platforms\android-35"))) {
    if (Test-Path -LiteralPath $SdkManager) {
        throw "Falta Android SDK Platform 35. Ejecute: `"$SdkManager`" `"platforms;android-35`" `"build-tools;35.0.0`" `"platform-tools`""
    }
    throw "Falta Android SDK Platform 35. Instálela desde Android Studio > SDK Manager."
}

$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
Push-Location $ProjectRoot
try {
    & .\gradlew.bat --no-daemon :app:assembleDebug
    if ($LASTEXITCODE -ne 0) { throw "Gradle terminó con código $LASTEXITCODE" }
    $Apk = Join-Path $ProjectRoot "app\build\outputs\apk\debug\app-debug.apk"
    Get-Item -LiteralPath $Apk | Select-Object FullName, Length, LastWriteTime
} finally {
    Pop-Location
}
