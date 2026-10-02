# =====================================================================
#  Qwen3-ASR - environment bootstrap
# =====================================================================
#  Creates .venv, installs the CPU runtime and all server dependencies.
#
#  Run from a normal PowerShell window:
#      cd E:\YOLO\qwen3-asr
#      powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#
#  Package source defaults to the official PyPI (pypi.org). If you are on
#  a network that cannot reach pypi.org (e.g. mainland China), pass a
#  domestic mirror explicitly:
#      .\setup_env.ps1 -Mirror "https://pypi.tuna.tsinghua.edu.cn/simple"
#
#  Model weights are downloaded by download_model.py from Hugging Face
#  (fallback: ModelScope via --source modelscope).
# =====================================================================

[CmdletBinding()]
param(
    [string]$PythonExe = "C:\Program Files\Python\python.exe",
    [string]$Mirror    = "https://pypi.org/simple",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

$Root    = $PSScriptRoot
$VenvDir = Join-Path $Root ".venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"
$TmpDir  = Join-Path $Root ".tmp"
$PipCache= Join-Path $TmpDir "pipcache"

Write-Host "== Qwen3-ASR environment setup ==" -ForegroundColor Cyan
Write-Host "project : $Root"
Write-Host "mirror  : $Mirror"

# ---------------------------------------------------------------- 1. python
if (-not (Test-Path $PythonExe)) {
    throw "Python interpreter not found: $PythonExe  (pass -PythonExe <path>)"
}
$pyVersion = & $PythonExe --version 2>&1
Write-Host "base python: $pyVersion ($PythonExe)"

# ---------------------------------------------------------------- 2. venv
if ($Recreate -and (Test-Path $VenvDir)) {
    Write-Host "removing existing venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvDir
}
if (-not (Test-Path $VenvPy)) {
    Write-Host "creating virtualenv..." -ForegroundColor Green
    New-Item -ItemType Directory -Force -Path $TmpDir, $PipCache | Out-Null
    & $PythonExe -m venv $VenvDir
}
if (-not (Test-Path $VenvPy)) { throw "virtualenv creation failed" }
Write-Host "venv python: $(& $VenvPy --version)"

# ---------------------------------------------------------------- 3. pip
Write-Host "upgrading pip / wheel / setuptools..." -ForegroundColor Green
& $VenvPy -m pip install --upgrade pip setuptools wheel `
    --index-url $Mirror --cache-dir $PipCache --disable-pip-version-check

# ------------------------------------------------- 4. qwen-asr (no deps)
# `sox` is declared by qwen-asr but never imported; installing it would try
# to build a C extension on Windows. Pull the package without dependencies
# and let requirements.txt supply the real ones.
Write-Host "installing qwen-asr (without its sox dependency)..." -ForegroundColor Green
& $VenvPy -m pip install --no-deps "qwen-asr" `
    --index-url $Mirror --cache-dir $PipCache --disable-pip-version-check

# ------------------------------------------------- 5. pinned requirements
Write-Host "installing runtime + server dependencies..." -ForegroundColor Green
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt") `
    --index-url $Mirror --cache-dir $PipCache --disable-pip-version-check

# ------------------------------------------------- 6. optional sox shim
# If some code path ever does `import sox`, this installs the pure-python
# wrapper only. It is harmless when unused and keeps imports from failing.
Write-Host "installing sox python wrapper (best effort, safe to fail)..." -ForegroundColor DarkGray
& $VenvPy -m pip install "sox" --index-url $Mirror --cache-dir $PipCache `
    --disable-pip-version-check 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  (skipped - not needed at runtime)" -ForegroundColor DarkGray
}

# ------------------------------------------------- 7. report
Write-Host ""
Write-Host "== installed versions ==" -ForegroundColor Cyan
& $VenvPy -c @"
import importlib.metadata as md
for p in ['torch','transformers','accelerate','librosa','soundfile','fastapi','uvicorn','modelscope','qwen-asr']:
    try:    print(f'  {p:14s} {md.version(p)}')
    except Exception: print(f'  {p:14s} (not installed)')
"@

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1) .venv\Scripts\python.exe download_model.py --model qwen3-asr-0.6b"
Write-Host "  2) .venv\Scripts\python.exe check_env.py"
Write-Host "  3) .venv\Scripts\python.exe server.py"
