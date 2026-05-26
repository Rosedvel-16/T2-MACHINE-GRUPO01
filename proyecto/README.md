# T2 — Sistema Inteligente Multiagente (Almacén Logístico)

Carpeta **única y ordenada** con todo el trabajo del proyecto: código, datos, imágenes, reportes y documentación.

## Estructura

```
proyecto/
├── README.md                 ← Este archivo
├── requirements.txt          ← Dependencias Python
├── run_dashboard.ps1         ← Lanzar Streamlit
├── src/                      ← Todo el código (Partes A–K)
│   ├── paths.py              ← Rutas centralizadas
│   ├── generador_datos.py    ← Parte A: simulación CSV
│   ├── analisis_senales.py   ← Parte B: scipy + gráficos
│   ├── vision_cv.py          ← Parte C: OpenCV
│   ├── sistema_experto.py    ← Partes D + E
│   ├── agentes.py            ← Parte F
│   ├── logica_proposicional.py
│   ├── logica_probabilidad.py
│   ├── concurrencia.py       ← Parte J
│   ├── optimizacion_despacho.py
│   └── streamlit_app.py      ← Dashboard unificado
├── data/
│   └── simulacion_almacen.csv
├── assets/
│   ├── imagenes_entrada/     ← 3 fotos del almacén (entrada OpenCV)
│   ├── imagenes_salida/      ← Resultados procesados + reporte_vision.json
│   └── reportes/             ← JSON de lógica, probabilidad, experto
└── docs/
    ├── guia_instalacion.md
    ├── informe_tecnico.md
    └── instruccion.txt
```

## Instalación

```powershell
cd proyecto
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución rápida

```powershell
# Dashboard Streamlit (recomendado)
.\run_dashboard.ps1

# O manualmente:
python -m streamlit run src\streamlit_app.py
```

## Scripts por módulo

Ejecutar siempre desde la carpeta `proyecto/`:

```powershell
python src\generador_datos.py
python src\analisis_senales.py
python src\vision_cv.py
python src\sistema_experto.py
python src\agentes.py
python src\concurrencia.py
python src\optimizacion_despacho.py
python src\logica_proposicional.py
python src\logica_probabilidad.py
```

## Nota sobre el resto del repositorio

Las carpetas `backend/` y `frontend/` en la raíz del repo son la versión anterior (dispersa). **Usa `proyecto/` como entregable principal.** El código aquí es el mismo, con rutas unificadas vía `src/paths.py`.
