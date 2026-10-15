param(
    [string]$SdkRoot = "$env:LOCALAPPDATA\Android\Sdk"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Destination = Join-Path $ProjectRoot "results"
$Adb = Join-Path $SdkRoot "platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $Adb)) { throw "adb no encontrado: $Adb" }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$Names = @(& $Adb shell run-as com.lmdyolo.benchmark ls files/results 2>$null)
if ($LASTEXITCODE -ne 0 -or $Names.Count -eq 0) { throw "No hay resultados accesibles en files/results" }
foreach ($Name in $Names) {
    $CleanName = $Name.Trim()
    if (-not $CleanName.EndsWith(".json")) { continue }
    $Text = (& $Adb exec-out run-as com.lmdyolo.benchmark cat "files/results/$CleanName") -join "`n"
    [IO.File]::WriteAllText((Join-Path $Destination $CleanName), $Text + "`n", [Text.UTF8Encoding]::new($false))
}
Get-ChildItem -LiteralPath $Destination -Filter "*.json" | Select-Object FullName, Length, LastWriteTime
