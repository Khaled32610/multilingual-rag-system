$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
$Streamlit = '.\.venv\Scripts\streamlit.exe'
if (!(Test-Path $Streamlit)) { throw 'Missing .venv. Run .\packaging\windows\install.ps1 first.' }
& $Streamlit run src/rag_system/ui/app.py --server.address 127.0.0.1 --server.port 8501
