"""
Rutas centralizadas del proyecto T2 — Almacén Logístico.
Todos los módulos importan desde aquí para funcionar desde cualquier directorio.
"""
from pathlib import Path

# Raíz del entregable (carpeta proyecto/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
IMAGENES_ENTRADA = ASSETS_DIR / "imagenes_entrada"
IMAGENES_SALIDA = ASSETS_DIR / "imagenes_salida"
REPORTES_DIR = ASSETS_DIR / "reportes"
DOCS_DIR = PROJECT_ROOT / "docs"

CSV_SENSORES = DATA_DIR / "simulacion_almacen.csv"
REPORTE_VISION = IMAGENES_SALIDA / "reporte_vision.json"
REPORTE_LOGICA = REPORTES_DIR / "reporte_logica.json"
REPORTE_PROBABILIDAD = REPORTES_DIR / "reporte_probabilidad.json"
REPORTE_SISTEMA_EXPERTO = REPORTES_DIR / "reporte_sistema_experto.json"


def asegurar_carpetas() -> None:
    """Crea directorios de salida si no existen."""
    for carpeta in (DATA_DIR, IMAGENES_ENTRADA, IMAGENES_SALIDA, REPORTES_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)
