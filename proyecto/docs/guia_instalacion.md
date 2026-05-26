# T2 Machine - Guia de instalacion y ejecucion

Esta guia resume lo necesario para ejecutar el backend con Streamlit y los
scripts principales del proyecto.

## Requisitos

- Windows 10/11
- Python 3.10+ (probado con 3.14)

## Instalacion (PowerShell)

Desde la raiz del repositorio:

```powershell
# (Opcional) crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
python -m pip install --upgrade pip
python -m pip install streamlit opencv-python numpy pandas matplotlib scipy
```

## Ejecutar el dashboard Streamlit

```powershell
# Ejecutar desde la raiz del repo para evitar problemas de site-packages
python -m streamlit run .\backend\ai_core\streamlit_app.py --server.headless true --browser.gatherUsageStats false
```

Abrir en el navegador: http://localhost:8501

## Scripts principales (opcional)

```powershell
# Parte A: generar datos simulados
python .\backend\data\generador_datos.py

# Parte B: analisis de senales
python .\backend\ai_core\analisis_senales.py

# Parte C: procesamiento de imagenes
python .\backend\ai_core\vision_cv.py

# Parte F: agentes (modo secuencial)
python .\backend\ai_core\agentes.py

# Parte J: concurrencia (threading)
python .\backend\ai_core\concurrencia.py

# Parte K: optimizacion de despacho
python .\backend\ai_core\optimizacion_despacho.py
```

## Notas de imagenes

- Las imagenes de entrada para vision se leen desde:
	`frontend/docs/imagenes_prueba/images_output`
- Los resultados de vision se escriben en:
	`backend/ai_core/images_output`