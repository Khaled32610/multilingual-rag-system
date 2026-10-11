$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
$Python = if ($env:PYTHON) { $env:PYTHON } else { 'py' }
if ($Python -eq 'py') { & py -3.11 -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ is required'" }
else { & $Python -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ is required'" }
if ($Python -eq 'py') { & py -3.11 -m venv .venv }
else { & $Python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
& .\.venv\Scripts\python.exe -m pip install -e '.[ui,reports]'
Write-Host 'Install complete. Run .\packaging\windows\start.ps1'
