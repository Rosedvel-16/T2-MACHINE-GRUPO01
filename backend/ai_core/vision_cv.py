"""
===============================================================================
PARTE C — PROCESAMIENTO BÁSICO DE IMÁGENES
Sistema Inteligente Multiagente para Almacén Logístico
Herramienta: OpenCV (cv2)
===============================================================================

Imágenes procesadas:
  1. Paquete dañado        → Detección de bordes (Canny) + umbralización adaptativa
  2. Paquete en buen estado → Umbralización simple (Otsu) + análisis de regiones
  3. Zona con obstrucción  → Detección de bordes + segmentación por color (HSV)

Cada imagen pasa por el siguiente pipeline:
  (a) Carga y validación
  (b) Conversión a escala de grises
  (c) Operación(es) específica(s) según contexto
  (d) Análisis cuantitativo de la operación
  (e) Guardado de resultados y reporte para el sistema inteligente
===============================================================================
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the shared frontend image folder as input source for Streamlit runs.
RUTA_ENTRADA = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "frontend", "docs", "imagenes_prueba", "images_output")
)
RUTA_SALIDA = os.path.join(BASE_DIR, "images_output")
os.makedirs(RUTA_SALIDA, exist_ok=True)

# ── Colores para reportes en consola ──────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def cargar_imagen(nombre_archivo: str) -> np.ndarray:
    """Carga una imagen y valida que exista."""
    ruta = os.path.join(RUTA_ENTRADA, nombre_archivo)
    img = cv2.imread(ruta)
    if img is None:
        # Fallback para rutas con caracteres Unicode en Windows
        try:
            data = np.fromfile(ruta, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        raise FileNotFoundError(
            f"No se pudo leer '{ruta}'. "
            "Verifica que el archivo exista y sea .jpg legible."
        )
    print(f"{GREEN}[CARGADA]{RESET} {nombre_archivo}  |  "
          f"shape={img.shape}  |  dtype={img.dtype}")
    return img

def guardar(img: np.ndarray, nombre: str) -> str:
    """Guarda una imagen procesada y retorna la ruta."""
    ruta = os.path.join(RUTA_SALIDA, nombre)
    ok = cv2.imwrite(ruta, img)
    if not ok:
        # Fallback para rutas con caracteres Unicode en Windows
        try:
            ext = os.path.splitext(ruta)[1]
            success, encoded = cv2.imencode(ext, img)
            if success:
                encoded.tofile(ruta)
                ok = True
        except Exception:
            ok = False
    if not ok:
        raise OSError(f"No se pudo guardar la imagen en: {ruta}")
    return ruta


def encabezado(titulo: str):
    sep = "═" * 70
    print(f"\n{CYAN}{BOLD}{sep}{RESET}")
    print(f"{CYAN}{BOLD}  {titulo}{RESET}")
    print(f"{CYAN}{BOLD}{sep}{RESET}\n")


def subtitulo(texto: str):
    print(f"\n{YELLOW}{BOLD}▶ {texto}{RESET}")


def listar_jpgs() -> list:
    """Lista archivos .jpg en la carpeta de entrada.

    Si existen archivos con sufijo "_a_original.jpg", se priorizan como entradas
    para evitar mezclar resultados intermedios.
    """
    archivos = [
        f for f in os.listdir(RUTA_ENTRADA)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(RUTA_ENTRADA, f))
    ]
    originales = [f for f in archivos if f.lower().endswith("_a_original.jpg")]
    return sorted(originales) if originales else sorted(archivos)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGEN 1 — PAQUETE DAÑADO
# Técnicas: Escala de grises → Suavizado Gaussiano → Canny (detección de bordes)
#            → Umbralización adaptativa
# ═══════════════════════════════════════════════════════════════════════════════

def procesar_paquete_danado(nombre_archivo: str) -> dict:
    """
    Objetivo: detectar bordes irregulares y zonas de daño en la superficie
    del paquete. Los bordes múltiples, fragmentados o en lugares inesperados
    indican deformaciones, manchas o roturas de la cinta de embalaje.
    """
    encabezado("IMAGEN 1 — PAQUETE DAÑADO  (Detección de bordes + Umbralización adaptativa)")

    # ── (a) Carga ──────────────────────────────────────────────────────────────
    original = cargar_imagen(nombre_archivo)
    guardar(original, "img1_a_original.jpg")

    # ── (b) Escala de grises ───────────────────────────────────────────────────
    subtitulo("(b) Conversión a escala de grises")
    gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    guardar(gris, "img1_b_grises.jpg")
    print(f"  Rango de intensidad: [{gris.min()}, {gris.max()}]  "
          f"| Media: {gris.mean():.1f}  | Std: {gris.std():.1f}")

    # ── (c-1) Suavizado Gaussiano (reduce ruido antes de Canny) ───────────────
    subtitulo("(c-1) Suavizado Gaussiano (kernel 5×5)")
    suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
    guardar(suavizado, "img1_c1_gaussiano.jpg")

    # ── (c-2) Detección de bordes — Canny ─────────────────────────────────────
    subtitulo("(c-2) Detección de bordes — Canny  (th1=50, th2=150)")
    bordes = cv2.Canny(suavizado, threshold1=50, threshold2=150)
    guardar(bordes, "img1_c2_canny.jpg")

    # Superposición roja sobre original para visualización clara
    overlay = original.copy()
    overlay[bordes > 0] = [0, 0, 220]   # Rojo BGR
    guardar(overlay, "img1_c2_canny_overlay.jpg")

    # ── (c-3) Umbralización adaptativa ────────────────────────────────────────
    subtitulo("(c-3) Umbralización adaptativa Gaussiana  (blockSize=11, C=2)")
    umbral = cv2.adaptiveThreshold(
        suavizado, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    guardar(umbral, "img1_c3_umbral_adaptativo.jpg")

    # ── (d) Análisis cuantitativo ──────────────────────────────────────────────
    subtitulo("(d) Análisis cuantitativo")
    px_total  = bordes.size
    px_borde  = int(np.count_nonzero(bordes))
    densidad  = px_borde / px_total * 100

    px_umbral = int(np.count_nonzero(umbral))
    pct_danio = px_umbral / px_total * 100

    # Contornos para estimar fragmentación
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contornos = len(contornos)

    print(f"  Píxeles de borde   : {px_borde:,} / {px_total:,}  ({densidad:.2f}%)")
    print(f"  Píxeles umbralados : {px_umbral:,}                 ({pct_danio:.2f}%)")
    print(f"  Nº de contornos    : {num_contornos}")

    # ── (e) Veredicto del agente de visión ─────────────────────────────────────
    subtitulo("(e) Diagnóstico para el sistema inteligente")
    danio_detectado = densidad > 5.0 or num_contornos > 80

    if danio_detectado:
        nivel = "ALTO" if densidad > 8.0 else "MEDIO"
        print(f"  {RED}[ALERTA] Daño probable en el paquete — Nivel {nivel}{RESET}")
        print(f"  → Densidad de bordes ({densidad:.1f}%) supera umbral normal (5%)")
        print(f"  → {num_contornos} contornos fragmentados indican superficie irregular")
        print(f"  → Acción recomendada: revisión manual inmediata")
    else:
        print(f"  {GREEN}[OK] Superficie dentro de parámetros normales{RESET}")

    resultado = {
        "imagen": nombre_archivo,
        "tecnicas": ["Escala de grises", "Gaussiano", "Canny", "Umbral adaptativo"],
        "px_borde": px_borde,
        "densidad_borde_pct": round(densidad, 2),
        "num_contornos": num_contornos,
        "px_umbralados_pct": round(pct_danio, 2),
        "danio_detectado": danio_detectado,
        "nivel_alerta": "ALTO" if densidad > 8.0 else ("MEDIO" if danio_detectado else "NINGUNO"),
        "archivos_salida": [
            "img1_b_grises.jpg", "img1_c1_gaussiano.jpg",
            "img1_c2_canny.jpg", "img1_c2_canny_overlay.jpg",
            "img1_c3_umbral_adaptativo.jpg"
        ]
    }
    return resultado


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGEN 2 — PAQUETE EN BUEN ESTADO
# Técnicas: Escala de grises → Umbralización Otsu → Análisis de regiones
# ═══════════════════════════════════════════════════════════════════════════════

def procesar_paquete_buen_estado(nombre_archivo: str) -> dict:
    """
    Objetivo: establecer una línea base de 'paquete correcto'. Otsu segmenta
    la imagen en dos clases (fondo/objeto) de forma óptima; si el paquete
    tiene pocas irregularidades, la región binarizada será compacta y regular.
    """
    encabezado("IMAGEN 2 — PAQUETE EN BUEN ESTADO  (Umbralización Otsu + morfología)")

    # ── (a) Carga ──────────────────────────────────────────────────────────────
    original = cargar_imagen(nombre_archivo)
    guardar(original, "img2_a_original.jpg")

    # ── (b) Escala de grises ───────────────────────────────────────────────────
    subtitulo("(b) Conversión a escala de grises")
    gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    guardar(gris, "img2_b_grises.jpg")
    print(f"  Media de intensidad: {gris.mean():.1f}  | Std: {gris.std():.1f}")

    # ── (c-1) Umbralización Otsu ───────────────────────────────────────────────
    subtitulo("(c-1) Umbralización Otsu (binarización óptima automática)")
    valor_otsu, umbral_otsu = cv2.threshold(
        gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    guardar(umbral_otsu, "img2_c1_otsu.jpg")
    print(f"  Umbral Otsu calculado: {valor_otsu:.1f}")

    # ── (c-2) Morfología: erosión + dilatación (apertura) para limpiar ruido ──
    subtitulo("(c-2) Apertura morfológica (kernel 3×3) — elimina ruido")
    kernel    = np.ones((3, 3), np.uint8)
    apertura  = cv2.morphologyEx(umbral_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    guardar(apertura, "img2_c2_apertura.jpg")

    # ── (c-3) Realce de contraste — CLAHE ────────────────────────────────────
    subtitulo("(c-3) Realce de contraste adaptativo CLAHE (clipLimit=2.0)")
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    realzado = clahe.apply(gris)
    guardar(realzado, "img2_c3_clahe.jpg")

    # ── (d) Análisis cuantitativo ──────────────────────────────────────────────
    subtitulo("(d) Análisis cuantitativo")
    px_total    = umbral_otsu.size
    px_objeto   = int(np.count_nonzero(apertura))
    pct_objeto  = px_objeto / px_total * 100

    contornos, _ = cv2.findContours(apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_grandes = [c for c in contornos if cv2.contourArea(c) > 500]

    # Compacidad del contorno más grande (indica regularidad)
    compacidad = 0.0
    if contornos_grandes:
        cnt_principal = max(contornos_grandes, key=cv2.contourArea)
        area      = cv2.contourArea(cnt_principal)
        perimetro = cv2.arcLength(cnt_principal, True)
        if perimetro > 0:
            compacidad = (4 * np.pi * area) / (perimetro ** 2)

    print(f"  Px objeto (apertura): {px_objeto:,}  ({pct_objeto:.1f}%)")
    print(f"  Contornos grandes   : {len(contornos_grandes)}")
    print(f"  Compacidad principal: {compacidad:.3f}  (1.0 = círculo perfecto; >0.5 = regular)")

    # ── (e) Veredicto ─────────────────────────────────────────────────────────
    subtitulo("(e) Diagnóstico para el sistema inteligente")
    estado_ok = compacidad > 0.3 and len(contornos_grandes) <= 5

    if estado_ok:
        print(f"  {GREEN}[OK] Paquete en buen estado — estructura compacta y regular{RESET}")
        print(f"  → Compacidad {compacidad:.3f} indica forma rectangular bien definida")
        print(f"  → Pocos contornos grandes: sin fragmentación anómala")
        print(f"  → Acción: apto para despacho directo")
    else:
        print(f"  {YELLOW}[ADVERTENCIA] Posibles irregularidades detectadas{RESET}")

    resultado = {
        "imagen": nombre_archivo,
        "tecnicas": ["Escala de grises", "Otsu", "Apertura morfológica", "CLAHE"],
        "umbral_otsu": round(valor_otsu, 1),
        "pct_objeto": round(pct_objeto, 2),
        "contornos_grandes": len(contornos_grandes),
        "compacidad": round(compacidad, 3),
        "estado_ok": estado_ok,
        "nivel_alerta": "NINGUNO" if estado_ok else "BAJO",
        "archivos_salida": [
            "img2_b_grises.jpg", "img2_c1_otsu.jpg",
            "img2_c2_apertura.jpg", "img2_c3_clahe.jpg"
        ]
    }
    return resultado


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGEN 3 — ZONA CON OBSTRUCCIÓN
# Técnicas: Escala de grises → Canny → Segmentación HSV por color de peligro
# ═══════════════════════════════════════════════════════════════════════════════

def procesar_zona_obstruida(nombre_archivo: str) -> dict:
    """
    Objetivo: detectar la presencia de objetos bloqueando pasillos y señales
    de advertencia. Se usa segmentación por color en espacio HSV para aislar
    las marcas de peligro (rojo/naranja) y Canny para medir la complejidad
    visual de la zona (más bordes = más desorden).
    """
    encabezado("IMAGEN 3 — ZONA CON OBSTRUCCIÓN  (Canny + Segmentación HSV)")

    # ── (a) Carga ──────────────────────────────────────────────────────────────
    original = cargar_imagen(nombre_archivo)
    guardar(original, "img3_a_original.jpg")

    # ── (b) Escala de grises ───────────────────────────────────────────────────
    subtitulo("(b) Conversión a escala de grises")
    gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    guardar(gris, "img3_b_grises.jpg")

    # ── (c-1) Detección de bordes Canny ───────────────────────────────────────
    subtitulo("(c-1) Canny para medir complejidad visual de la zona")
    suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes    = cv2.Canny(suavizado, 30, 120)
    guardar(bordes, "img3_c1_canny.jpg")

    overlay_bordes = original.copy()
    overlay_bordes[bordes > 0] = [0, 255, 255]  # Amarillo
    guardar(overlay_bordes, "img3_c1_canny_overlay.jpg")

    # ── (c-2) Segmentación por color HSV — detecta señales de peligro ─────────
    subtitulo("(c-2) Segmentación HSV — detectar señal de advertencia (naranja/amarillo)")
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # Rango naranja/amarillo (señal de advertencia triangular)
    bajo1  = np.array([15, 100, 100])
    alto1  = np.array([35, 255, 255])
    mascara_adv = cv2.inRange(hsv, bajo1, alto1)

    # Rango rojo (líneas de peligro en el suelo) — dos rangos en HSV
    bajo_r1 = np.array([0,  120, 100])
    alto_r1 = np.array([10, 255, 255])
    bajo_r2 = np.array([165, 120, 100])
    alto_r2 = np.array([180, 255, 255])
    mascara_rojo = cv2.inRange(hsv, bajo_r1, alto_r1) | cv2.inRange(hsv, bajo_r2, alto_r2)

    mascara_total = cv2.bitwise_or(mascara_adv, mascara_rojo)

    # Operaciones morfológicas para cerrar huecos
    kernel  = np.ones((5, 5), np.uint8)
    mascara_clean = cv2.morphologyEx(mascara_total, cv2.MORPH_CLOSE, kernel)
    guardar(mascara_clean, "img3_c2_mascara_peligro.jpg")

    # Aplicar máscara sobre original
    zona_peligro = cv2.bitwise_and(original, original, mask=mascara_clean)
    guardar(zona_peligro, "img3_c2_zona_peligro.jpg")

    # ── (c-3) Dilatación sobre bordes para resaltar objetos en pasillo ─────────
    subtitulo("(c-3) Dilatación morfológica — resaltar objetos en pasillo")
    dilatado = cv2.dilate(bordes, np.ones((3,3), np.uint8), iterations=2)
    guardar(dilatado, "img3_c3_dilatado.jpg")

    # ── (d) Análisis cuantitativo ──────────────────────────────────────────────
    subtitulo("(d) Análisis cuantitativo")
    px_total      = bordes.size
    px_borde      = int(np.count_nonzero(bordes))
    densidad_borde = px_borde / px_total * 100

    px_peligro     = int(np.count_nonzero(mascara_clean))
    pct_peligro    = px_peligro / px_total * 100

    contornos_obs, _ = cv2.findContours(mascara_clean, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    num_zonas_peligro = len(contornos_obs)

    print(f"  Densidad de bordes         : {densidad_borde:.2f}%")
    print(f"  Px señales de peligro      : {px_peligro:,}  ({pct_peligro:.2f}%)")
    print(f"  Zonas de peligro detectadas: {num_zonas_peligro}")

    # ── (e) Veredicto ─────────────────────────────────────────────────────────
    subtitulo("(e) Diagnóstico para el sistema inteligente")
    obstruccion_detectada = densidad_borde > 6.0 or num_zonas_peligro >= 1

    if obstruccion_detectada:
        nivel = "CRITICO" if (densidad_borde > 10.0 and num_zonas_peligro > 2) else "ALTO"
        print(f"  {RED}[ALERTA] Obstrucción detectada en zona — Nivel {nivel}{RESET}")
        print(f"  → Densidad de bordes {densidad_borde:.1f}% supera umbral de zona despejada (<4%)")
        print(f"  → {num_zonas_peligro} zona(s) con marcas de peligro identificadas")
        print(f"  → Acción: bloquear ruta, notificar operador, reubicar carga")
    else:
        print(f"  {GREEN}[OK] Zona despejada — circulación libre{RESET}")

    resultado = {
        "imagen": nombre_archivo,
        "tecnicas": ["Escala de grises", "Canny", "Segmentación HSV", "Morfología"],
        "densidad_borde_pct": round(densidad_borde, 2),
        "px_peligro_pct": round(pct_peligro, 2),
        "zonas_peligro": num_zonas_peligro,
        "obstruccion_detectada": obstruccion_detectada,
        "nivel_alerta": "CRITICO" if (densidad_borde > 10.0 and num_zonas_peligro > 2) else
                        ("ALTO" if obstruccion_detectada else "NINGUNO"),
        "archivos_salida": [
            "img3_b_grises.jpg", "img3_c1_canny.jpg", "img3_c1_canny_overlay.jpg",
            "img3_c2_mascara_peligro.jpg", "img3_c2_zona_peligro.jpg",
            "img3_c3_dilatado.jpg"
        ]
    }
    return resultado


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Ejecuta el pipeline completo y genera reporte JSON
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'='*70}")
    print("  PARTE C — PROCESAMIENTO BÁSICO DE IMÁGENES")
    print(f"  Sistema Inteligente Multiagente — Almacén Logístico")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{RESET}\n")

    resultados = []

    jpgs = listar_jpgs()
    if len(jpgs) < 3:
        raise FileNotFoundError(
            f"Se requieren 3 imágenes .jpg en {RUTA_ENTRADA}. "
            f"Encontradas: {len(jpgs)}"
        )

    # Se usan las tres primeras .jpg en orden alfabético
    img_danado, img_bueno, img_obstruida = jpgs[:3]
    print(f"{YELLOW}Usando imágenes (orden alfabético):{RESET}")
    print(f"  1) Paquete dañado     -> {img_danado}")
    print(f"  2) Paquete buen estado-> {img_bueno}")
    print(f"  3) Zona obstruida     -> {img_obstruida}")

    # Procesar las tres imágenes
    resultados.append(procesar_paquete_danado(img_danado))
    resultados.append(procesar_paquete_buen_estado(img_bueno))
    resultados.append(procesar_zona_obstruida(img_obstruida))

    # ── Resumen global ─────────────────────────────────────────────────────────
    encabezado("RESUMEN GLOBAL — PARTE C")

    alertas = {
        "CRITICO": [r for r in resultados if r.get("nivel_alerta") == "CRITICO"],
        "ALTO"   : [r for r in resultados if r.get("nivel_alerta") == "ALTO"],
        "MEDIO"  : [r for r in resultados if r.get("nivel_alerta") == "MEDIO"],
        "BAJO"   : [r for r in resultados if r.get("nivel_alerta") == "BAJO"],
        "NINGUNO": [r for r in resultados if r.get("nivel_alerta") == "NINGUNO"],
    }

    for nivel, lista in alertas.items():
        color = RED if nivel in ("CRITICO","ALTO") else (YELLOW if nivel=="MEDIO" else GREEN)
        if lista:
            for r in lista:
                print(f"  {color}[{nivel:8s}]{RESET}  {r['imagen']}")

    print(f"\n  Imágenes guardadas en: {os.path.abspath(RUTA_SALIDA)}/")
    archivos_total = sum(len(r["archivos_salida"]) for r in resultados)
    print(f"  Total archivos procesados: {archivos_total} imágenes de salida")

    # ── Guardar reporte JSON ───────────────────────────────────────────────────
    reporte = {
        "modulo": "Parte C — Procesamiento de imágenes (vision_cv.py)",
        "timestamp": datetime.now().isoformat(),
        "imagenes_procesadas": len(resultados),
        "resultados": resultados
    }
    ruta_reporte = os.path.join(RUTA_SALIDA, "reporte_vision.json")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print(f"\n  {GREEN}Reporte JSON guardado: {ruta_reporte}{RESET}")
    print(f"\n{BOLD}{'='*70}")
    print("  Parte C completada exitosamente.")
    print(f"{'='*70}{RESET}\n")

    return reporte


if __name__ == "__main__":
    main()
