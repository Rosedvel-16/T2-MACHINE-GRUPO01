# Ejecuta el dashboard desde la carpeta proyecto/
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m streamlit run src\streamlit_app.py --server.headless true --browser.gatherUsageStats false
