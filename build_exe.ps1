param(
    [string]$PythonExe = "D:\App\anaconda\anaconda\envs\psychopy_pip_env\python.exe",
    [string]$EntryScript = "grad_cpt.py",
    [string]$AppName = "CPTManager",
    [switch]$CleanFirst
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $EntryScript)) {
    throw "Entry script not found: $EntryScript"
}

Write-Host "Using Python: $PythonExe"

& $PythonExe -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller not found. Installing..."
    & $PythonExe -m pip install pyinstaller
}

if ($CleanFirst) {
    if (Test-Path ".\dist") { Remove-Item ".\dist" -Recurse -Force }
    if (Test-Path ".\build") { Remove-Item ".\build" -Recurse -Force }
    if (Test-Path ".\$AppName.spec") { Remove-Item ".\$AppName.spec" -Force }
}

$args = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--onedir",
    "--name", $AppName,
    "--collect-all", "psychopy",
    "--collect-all", "pandas",
    "--collect-all", "numpy",
    "--collect-all", "scipy",
    "--collect-all", "openpyxl",
    "--collect-all", "pyglet",
    "--hidden-import", "wx",
    "--hidden-import", "wx.grid",
    $EntryScript
)

Write-Host "Running PyInstaller..."
& $PythonExe @args

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

$outDir = Join-Path (Join-Path (Get-Location) "dist") $AppName
if (-not (Test-Path $outDir)) {
    throw "Build finished but output folder not found: $outDir"
}

if (Test-Path ".\supplementary_info.zip") {
    Copy-Item ".\supplementary_info.zip" (Join-Path $outDir "supplementary_info.zip") -Force
    Write-Host "Copied supplementary_info.zip to output folder."
}

Write-Host ""
Write-Host "Build done."
Write-Host "Executable folder: $outDir"
Write-Host "Main executable: $(Join-Path $outDir ($AppName + '.exe'))"

