"""
===============================================================================
PARTE D — SISTEMA EXPERTO BASADO EN REGLAS
PARTE E — REPRESENTACIÓN DEL CONOCIMIENTO
Sistema Inteligente Multiagente para Almacén Logístico
Autor: Persona 3 (Arquitecto de la inteligencia del sistema)
===============================================================================

Este módulo implementa el "cerebro" simbólico del almacén:

  PARTE D — Sistema experto:
    1. Base de hechos (≥ 8 hechos)        ─→ HECHOS_BASE / clase Hecho
    2. Base de reglas if/then (≥ 6 reglas) ─→ REGLAS / clase Regla
    3. Motor de inferencia hacia adelante  ─→ MotorExperto.inferir()
    4. Consultas con explicación (proof trace)

  PARTE E — Representación del conocimiento:
    1. Hechos como dataclasses con metadatos (origen, confianza, timestamp)
    2. Reglas como objetos con antecedentes, consecuente, prioridad y acción
    3. Tabla de conocimiento (matriz hechos ↔ reglas)
    4. Diccionario de relaciones causales entre eventos

El sistema lee los CSV de sensores (Parte A — Gaby), los reportes de visión
(Parte C — Cano) y devuelve un diagnóstico estructurado consumible por los
agentes (Parte F) y el dashboard Streamlit.
===============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import sys

# Forzar UTF-8 en la salida (necesario para Windows / cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Colores consola (mismo estilo del resto del proyecto) ─────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE E — REPRESENTACIÓN DEL CONOCIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
#
# El conocimiento del almacén se modela con tres elementos:
#   1. HECHOS        : afirmaciones atómicas sobre el estado del mundo
#   2. REGLAS        : implicaciones lógicas if/then con acciones asociadas
#   3. CONCLUSIONES  : nuevos hechos derivados por inferencia
#
# Cada hecho lleva metadatos (origen del sensor, confianza, marca de tiempo)
# para que el sistema pueda razonar bajo incertidumbre y explicar sus
# decisiones (auditoría).
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Hecho:
    """Representa un hecho atómico del dominio del almacén."""
    nombre      : str                    # Identificador único (ej. "temperatura_alta")
    descripcion : str                    # Significado en lenguaje natural
    dominio     : str                    # sensores | vision | logistica | ambiental
    activo      : bool = False           # ¿Está presente en el estado actual?
    confianza   : float = 1.0            # Nivel de confianza [0,1] (Parte H compat.)
    origen      : str = "desconocido"    # Qué módulo/agente lo introdujo
    timestamp   : str = field(default_factory=lambda: datetime.now().isoformat())

    def __repr__(self) -> str:
        estado = f"{GREEN}✓{RESET}" if self.activo else f"{DIM}✗{RESET}"
        return f"{estado} {self.nombre} (conf={self.confianza:.0%})"


@dataclass
class Regla:
    """Representa una regla de producción IF (antecedentes) THEN (consecuente)."""
    id              : str
    descripcion     : str
    antecedentes    : List[str]                # Nombres de hechos requeridos
    consecuente     : str                      # Hecho nuevo que se concluye
    accion          : str                      # Acción operativa recomendada
    prioridad       : int = 1                  # 1=baja, 2=media, 3=alta, 4=crítica
    condicion_extra : Optional[Callable[[Dict[str, Hecho]], bool]] = None
    # condicion_extra: lambda opcional para combinaciones complejas (AND/OR)

    def evaluar(self, base_hechos: Dict[str, Hecho]) -> bool:
        """
        Devuelve True si TODOS los antecedentes están activos en la base de
        hechos y la condición extra (si existe) también se cumple.
        """
        todos_activos = all(
            base_hechos.get(a, Hecho(a, "", "")).activo
            for a in self.antecedentes
        )
        if not todos_activos:
            return False
        if self.condicion_extra is not None:
            return self.condicion_extra(base_hechos)
        return True

    def confianza_combinada(self, base_hechos: Dict[str, Hecho]) -> float:
        """
        Combina la confianza de los antecedentes (regla del mínimo, típica en
        sistemas expertos clásicos tipo MYCIN).
        """
        confianzas = [base_hechos[a].confianza
                      for a in self.antecedentes if a in base_hechos]
        return min(confianzas) if confianzas else 1.0


# ── CATÁLOGO DE HECHOS (≥ 8, requisito Parte D) ────────────────────────────────
HECHOS_BASE: Dict[str, Hecho] = {h.nombre: h for h in [
    Hecho("temperatura_alta",       "Temperatura interna del almacén supera 30 °C",        "ambiental"),
    Hecho("humedad_excesiva",       "Humedad relativa por encima del 80%",                 "ambiental"),
    Hecho("vibracion_anomala",      "Vibración en fajas/estanterías supera el umbral seguro", "sensores"),
    Hecho("zona_saturada",          "Ocupación de la zona de carga supera el 90%",         "logistica"),
    Hecho("paquete_danado",         "La inspección por visión detecta daño en el paquete", "vision"),
    Hecho("ruta_obstruida",         "La ruta principal del pasillo está bloqueada",        "vision"),
    Hecho("prioridad_alta_pedido",  "Existe un pedido con prioridad VIP en cola",          "logistica"),
    Hecho("sensor_fallido",         "Un sensor lleva varios ciclos sin reportar lectura",  "sensores"),
    Hecho("zona_libre",             "La zona de despacho tiene ocupación < 40%",           "logistica"),
    Hecho("mantenimiento_diferido", "No se ha realizado mantenimiento preventivo a tiempo","operativo"),
]}


# ── CATÁLOGO DE REGLAS (≥ 6, requisito Parte D) ────────────────────────────────
REGLAS: List[Regla] = [

    Regla(
        id="R-01",
        descripcion="Si la vibración es anómala → existe riesgo mecánico.",
        antecedentes=["vibracion_anomala"],
        consecuente="riesgo_mecanico",
        accion="Inspeccionar fajas y rodamientos; programar parada preventiva.",
        prioridad=3,
    ),

    Regla(
        id="R-02",
        descripcion="Si temperatura alta Y humedad excesiva → alerta ambiental.",
        antecedentes=["temperatura_alta", "humedad_excesiva"],
        consecuente="alerta_ambiental",
        accion="Encender HVAC, ventilación forzada y notificar a supervisor.",
        prioridad=3,
    ),

    Regla(
        id="R-03",
        descripcion="Si un paquete está dañado → requiere revisión manual.",
        antecedentes=["paquete_danado"],
        consecuente="revision_manual_paquete",
        accion="Retirar paquete de la línea y derivar a estación de inspección.",
        prioridad=2,
    ),

    Regla(
        id="R-04",
        descripcion="Si zona saturada Y ruta obstruida → reubicar carga.",
        antecedentes=["zona_saturada", "ruta_obstruida"],
        consecuente="reubicar_carga",
        accion="Asignar montacarga a zona auxiliar y liberar pasillo principal.",
        prioridad=3,
    ),

    Regla(
        id="R-05",
        descripcion="Si pedido prioritario Y zona libre → despacho inmediato.",
        antecedentes=["prioridad_alta_pedido", "zona_libre"],
        consecuente="despacho_inmediato",
        accion="Asignar el pedido VIP al próximo vehículo disponible.",
        prioridad=2,
    ),

    Regla(
        id="R-06",
        descripcion="Si riesgo mecánico Y mantenimiento diferido → parada de emergencia.",
        antecedentes=["riesgo_mecanico", "mantenimiento_diferido"],
        consecuente="parada_emergencia",
        accion="Detener operaciones, aislar zona y activar protocolo de seguridad.",
        prioridad=4,
    ),

    Regla(
        id="R-07",
        descripcion="Si sensor fallido → marcar datos como no confiables.",
        antecedentes=["sensor_fallido"],
        consecuente="datos_no_confiables",
        accion="Activar sensor redundante y enviar técnico al nodo afectado.",
        prioridad=2,
    ),

    Regla(
        id="R-08",
        descripcion="Si alerta ambiental Y zona saturada → riesgo de pérdida de mercancía.",
        antecedentes=["alerta_ambiental", "zona_saturada"],
        consecuente="riesgo_mercancia",
        accion="Reubicar mercancía sensible a zona climatizada.",
        prioridad=3,
    ),
]


# ── TABLA DE CONOCIMIENTO (matriz hechos × reglas) — Parte E ──────────────────
def construir_tabla_conocimiento() -> List[Dict[str, Any]]:
    """
    Devuelve la matriz que vincula cada hecho con las reglas en las que
    interviene como antecedente. Esta tabla es la EVIDENCIA visual pedida
    por la Parte E.
    """
    tabla = []
    for nombre, hecho in HECHOS_BASE.items():
        reglas_relacionadas = [r.id for r in REGLAS if nombre in r.antecedentes]
        tabla.append({
            "hecho"        : nombre,
            "descripcion"  : hecho.descripcion,
            "dominio"      : hecho.dominio,
            "reglas_uso"   : reglas_relacionadas,
            "es_conclusion": any(r.consecuente == nombre for r in REGLAS),
        })
    return tabla


# ── RELACIONES CAUSALES — Parte E ─────────────────────────────────────────────
# Diccionario que documenta cadenas de razonamiento típicas en el dominio.
RELACIONES_CAUSALES: Dict[str, List[str]] = {
    "vibracion_anomala"     : ["riesgo_mecanico", "parada_emergencia"],
    "temperatura_alta"      : ["alerta_ambiental", "riesgo_mercancia"],
    "humedad_excesiva"      : ["alerta_ambiental", "riesgo_mercancia"],
    "paquete_danado"        : ["revision_manual_paquete"],
    "zona_saturada"         : ["reubicar_carga", "riesgo_mercancia"],
    "ruta_obstruida"        : ["reubicar_carga"],
    "prioridad_alta_pedido" : ["despacho_inmediato"],
    "sensor_fallido"        : ["datos_no_confiables"],
    "mantenimiento_diferido": ["parada_emergencia"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# PARTE D — MOTOR DE INFERENCIA HACIA ADELANTE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Conclusion:
    """Resultado de aplicar una regla."""
    regla_id    : str
    hecho_nuevo : str
    accion      : str
    prioridad   : int
    confianza   : float
    antecedentes: List[str]
    timestamp   : str = field(default_factory=lambda: datetime.now().isoformat())


class MotorExperto:
    """
    Motor de inferencia hacia adelante (forward chaining).

    Algoritmo:
      1. Tomar la base de hechos inicial.
      2. Recorrer todas las reglas; si una se dispara, añadir su consecuente
         a la base de hechos como un nuevo hecho activo.
      3. Repetir hasta que no se generen más conclusiones nuevas (punto fijo).
      4. Devolver la lista ordenada por prioridad de las acciones recomendadas.

    Ventajas: razonamiento explícito, trazable, fácil de auditar y de
    explicar al usuario final (consultas con explicación).
    """

    def __init__(self,
                 hechos: Optional[Dict[str, Hecho]] = None,
                 reglas: Optional[List[Regla]] = None):
        # Copia profunda manual de los hechos para no mutar el catálogo base.
        self.base_hechos: Dict[str, Hecho] = {
            n: Hecho(h.nombre, h.descripcion, h.dominio,
                     h.activo, h.confianza, h.origen, h.timestamp)
            for n, h in (hechos or HECHOS_BASE).items()
        }
        self.reglas        = reglas or REGLAS
        self.conclusiones  : List[Conclusion] = []
        self.traza         : List[str] = []   # Histórico de razonamiento

    # ── API pública ──────────────────────────────────────────────────────────
    def asertar_hecho(self, nombre: str, confianza: float = 1.0,
                       origen: str = "manual") -> None:
        """Marca un hecho como activo (lo añade a la base si no existe)."""
        if nombre not in self.base_hechos:
            self.base_hechos[nombre] = Hecho(nombre, "(hecho inferido)", "derivado")
        h = self.base_hechos[nombre]
        h.activo    = True
        h.confianza = confianza
        h.origen    = origen
        h.timestamp = datetime.now().isoformat()
        self.traza.append(
            f"[ASERTAR] {nombre} → activo (conf={confianza:.0%}, origen={origen})"
        )

    def retraer_hecho(self, nombre: str) -> None:
        """Marca un hecho como inactivo."""
        if nombre in self.base_hechos:
            self.base_hechos[nombre].activo = False
            self.traza.append(f"[RETRAER] {nombre} → inactivo")

    def inferir(self, max_iter: int = 10) -> List[Conclusion]:
        """
        Ejecuta forward chaining hasta el punto fijo.
        Devuelve la lista de conclusiones disparadas (ordenadas por prioridad).
        """
        self.conclusiones.clear()
        ya_disparadas: Set[str] = set()

        for iteracion in range(1, max_iter + 1):
            disparadas_en_ciclo = 0

            for regla in self.reglas:
                if regla.id in ya_disparadas:
                    continue

                if regla.evaluar(self.base_hechos):
                    conf = regla.confianza_combinada(self.base_hechos)
                    conclusion = Conclusion(
                        regla_id    = regla.id,
                        hecho_nuevo = regla.consecuente,
                        accion      = regla.accion,
                        prioridad   = regla.prioridad,
                        confianza   = conf,
                        antecedentes= list(regla.antecedentes),
                    )
                    self.conclusiones.append(conclusion)
                    self.asertar_hecho(regla.consecuente,
                                        confianza=conf,
                                        origen=f"regla {regla.id}")
                    self.traza.append(
                        f"[CICLO {iteracion}] {regla.id} disparada: "
                        f"({' ∧ '.join(regla.antecedentes)}) → "
                        f"{regla.consecuente}  (conf={conf:.0%})"
                    )
                    ya_disparadas.add(regla.id)
                    disparadas_en_ciclo += 1

            if disparadas_en_ciclo == 0:
                self.traza.append(
                    f"[CICLO {iteracion}] No hay más reglas que disparar — fin."
                )
                break

        self.conclusiones.sort(key=lambda c: (-c.prioridad, -c.confianza))
        return self.conclusiones

    # ── Consultas y explicaciones ────────────────────────────────────────────
    def explicar(self, nombre_hecho: str) -> List[str]:
        """
        Devuelve la cadena de reglas que llevaron a concluir un hecho dado.
        Útil para mostrar al usuario "por qué" el sistema llegó a tal alerta.
        """
        explicacion: List[str] = []
        objetivo = nombre_hecho
        visitados: Set[str] = set()

        while objetivo and objetivo not in visitados:
            visitados.add(objetivo)
            origen = next(
                (c for c in self.conclusiones if c.hecho_nuevo == objetivo),
                None
            )
            if origen is None:
                if objetivo in self.base_hechos and self.base_hechos[objetivo].activo:
                    h = self.base_hechos[objetivo]
                    explicacion.append(
                        f"• {objetivo}: hecho directo de '{h.origen}' (conf {h.confianza:.0%})"
                    )
                break

            explicacion.append(
                f"• Regla {origen.regla_id}: {' ∧ '.join(origen.antecedentes)} "
                f"→ {origen.hecho_nuevo}  (conf {origen.confianza:.0%})"
            )
            objetivo = origen.antecedentes[0] if origen.antecedentes else None

        return explicacion

    def consultar(self, nombre_hecho: str) -> Dict[str, Any]:
        """Consulta puntual sobre el estado de un hecho/conclusión."""
        if nombre_hecho not in self.base_hechos:
            return {"existe": False}

        h = self.base_hechos[nombre_hecho]
        return {
            "existe"     : True,
            "activo"     : h.activo,
            "confianza"  : h.confianza,
            "origen"     : h.origen,
            "descripcion": h.descripcion,
            "explicacion": self.explicar(nombre_hecho) if h.activo else [],
        }

    def hechos_activos(self) -> List[Hecho]:
        """Devuelve la lista de hechos actualmente activos."""
        return [h for h in self.base_hechos.values() if h.activo]


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRACIÓN CON SEÑALES (Parte B) E IMÁGENES (Parte C)
# ═══════════════════════════════════════════════════════════════════════════════
#
# El sistema experto NO repite el análisis numérico: traduce las salidas
# numéricas de los otros módulos en hechos lógicos. Estos helpers son los
# que usarán los agentes de la Parte F.
# ═══════════════════════════════════════════════════════════════════════════════

def hechos_desde_sensores(df_row: Dict[str, float],
                            umbrales: Optional[Dict[str, float]] = None
                            ) -> List[Tuple[str, float, str]]:
    """
    Traduce una fila del CSV de sensores (Parte A) en una lista de hechos
    a asertar en el motor.

    df_row: dict con claves temperatura, humedad, vibracion, ocupacion.
    Devuelve: lista de tuplas (nombre_hecho, confianza, origen).
    """
    u = umbrales or {"temperatura": 30, "humedad": 75, "vibracion": 20, "ocupacion": 90}
    hechos: List[Tuple[str, float, str]] = []

    if df_row.get("temperatura", 0) > u["temperatura"]:
        exceso = df_row["temperatura"] - u["temperatura"]
        conf = min(0.99, 0.70 + exceso * 0.03)
        hechos.append(("temperatura_alta", conf, "sensor:temp"))

    if df_row.get("humedad", 0) > u["humedad"]:
        conf = min(0.99, 0.65 + (df_row["humedad"] - u["humedad"]) * 0.02)
        hechos.append(("humedad_excesiva", conf, "sensor:hum"))

    if df_row.get("vibracion", 0) > u["vibracion"]:
        conf = min(0.99, 0.75 + (df_row["vibracion"] - u["vibracion"]) * 0.02)
        hechos.append(("vibracion_anomala", conf, "sensor:vib"))

    if df_row.get("ocupacion", 0) > u["ocupacion"]:
        conf = min(0.99, 0.70 + (df_row["ocupacion"] - u["ocupacion"]) * 0.03)
        hechos.append(("zona_saturada", conf, "sensor:ocup"))

    return hechos


def hechos_desde_vision(reporte_vision: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    Traduce el reporte JSON de vision_cv.py (Parte C) en hechos lógicos.
    Acepta el dict que vision_cv.main() devuelve.
    """
    hechos: List[Tuple[str, float, str]] = []
    for r in reporte_vision.get("resultados", []):
        nombre = r.get("imagen", "").lower()
        nivel  = r.get("nivel_alerta", "NINGUNO")

        # Daño en paquete
        if r.get("danio_detectado"):
            conf = 0.88 if nivel == "ALTO" else 0.75
            hechos.append(("paquete_danado", conf, f"vision:{nombre}"))

        # Obstrucción de ruta
        if r.get("obstruccion_detectada"):
            conf = 0.85 if nivel in ("ALTO", "CRITICO") else 0.70
            hechos.append(("ruta_obstruida", conf, f"vision:{nombre}"))
    return hechos


# ═══════════════════════════════════════════════════════════════════════════════
# ESCENARIOS DE PRUEBA (sirven al __main__ y al dashboard)
# ═══════════════════════════════════════════════════════════════════════════════

ESCENARIOS_EXPERTO: Dict[str, List[str]] = {

    "ESC-A — Operación normal": [
        "zona_libre",
    ],

    "ESC-B — Falla mecánica con mantenimiento diferido": [
        "vibracion_anomala",
        "mantenimiento_diferido",
    ],

    "ESC-C — Alerta ambiental crítica": [
        "temperatura_alta",
        "humedad_excesiva",
        "zona_saturada",
    ],

    "ESC-D — Paquete dañado + pedido VIP": [
        "paquete_danado",
        "prioridad_alta_pedido",
        "zona_libre",
    ],

    "ESC-E — Crisis múltiple": [
        "vibracion_anomala", "mantenimiento_diferido",
        "temperatura_alta", "humedad_excesiva",
        "zona_saturada", "ruta_obstruida",
        "paquete_danado",
    ],

    "ESC-F — Sensor caído + operación en marcha": [
        "sensor_fallido",
        "prioridad_alta_pedido",
        "zona_libre",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PRESENTACIÓN (consola)
# ═══════════════════════════════════════════════════════════════════════════════

def imprimir_base_conocimiento():
    """Muestra hechos, reglas y tabla — evidencia para la Parte E."""
    print(f"\n{CYAN}{BOLD}{'─'*70}")
    print("  BASE DE HECHOS (≥ 8 hechos)")
    print(f"{'─'*70}{RESET}")
    print(f"  {'NOMBRE':<26} {'DOMINIO':<12} DESCRIPCIÓN")
    print(f"  {'─'*24}  {'─'*10}  {'─'*30}")
    for h in HECHOS_BASE.values():
        print(f"  {BOLD}{h.nombre:<26}{RESET} {DIM}{h.dominio:<12}{RESET} {h.descripcion}")

    print(f"\n{CYAN}{BOLD}{'─'*70}")
    print("  BASE DE REGLAS (≥ 6 reglas)")
    print(f"{'─'*70}{RESET}")
    for r in REGLAS:
        color_pri = RED if r.prioridad == 4 else (YELLOW if r.prioridad == 3 else GREEN)
        print(f"\n  {BOLD}[{r.id}]{RESET} (prioridad {color_pri}{r.prioridad}{RESET})")
        print(f"     IF   {' ∧ '.join(r.antecedentes)}")
        print(f"     THEN {BOLD}{r.consecuente}{RESET}")
        print(f"     ACCIÓN: {DIM}{r.accion}{RESET}")


def imprimir_tabla_conocimiento():
    """Imprime la matriz hechos ↔ reglas."""
    tabla = construir_tabla_conocimiento()
    print(f"\n{CYAN}{BOLD}{'─'*70}")
    print("  TABLA DE CONOCIMIENTO (matriz hechos × reglas)")
    print(f"{'─'*70}{RESET}")
    print(f"  {'HECHO':<26} {'DOMINIO':<12} {'TIPO':<10} REGLAS")
    print(f"  {'─'*24}  {'─'*10}  {'─'*8}  {'─'*16}")
    for fila in tabla:
        tipo = "derivado" if fila["es_conclusion"] else "primitivo"
        reglas_str = ", ".join(fila["reglas_uso"]) if fila["reglas_uso"] else "—"
        color_tipo = YELLOW if tipo == "derivado" else GREEN
        print(f"  {BOLD}{fila['hecho']:<26}{RESET} "
              f"{DIM}{fila['dominio']:<12}{RESET} "
              f"{color_tipo}{tipo:<10}{RESET}{reglas_str}")


def imprimir_resultado_escenario(nombre: str, motor: MotorExperto):
    """Muestra de forma clara el resultado de inferencia de un escenario."""
    print(f"\n{YELLOW}{BOLD}  ▶ {nombre}{RESET}")
    print(f"  {'─'*65}")

    activos = motor.hechos_activos()
    primitivos = [h for h in activos
                  if not any(r.consecuente == h.nombre for r in REGLAS)]
    derivados  = [h for h in activos
                  if any(r.consecuente == h.nombre for r in REGLAS)]

    print(f"  Hechos asertados ({len(primitivos)}): "
          f"{', '.join(h.nombre for h in primitivos)}")

    if motor.conclusiones:
        print(f"\n  {BOLD}Conclusiones derivadas (ordenadas por prioridad):{RESET}")
        for c in motor.conclusiones:
            col = RED if c.prioridad == 4 else (YELLOW if c.prioridad == 3 else GREEN)
            print(f"    {col}[P{c.prioridad}]{RESET} {c.hecho_nuevo:<26} "
                  f"(conf {c.confianza:.0%})  ← {c.regla_id}")
            print(f"         → {DIM}{c.accion}{RESET}")
    else:
        print(f"\n  {GREEN}✓ Sin alertas — operación normal.{RESET}")

    # Mostrar explicación de la conclusión más crítica
    if motor.conclusiones:
        critica = motor.conclusiones[0]
        print(f"\n  {CYAN}Explicación de '{critica.hecho_nuevo}':{RESET}")
        for linea in motor.explicar(critica.hecho_nuevo):
            print(f"    {linea}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Ejecuta los escenarios y guarda el reporte JSON
# ═══════════════════════════════════════════════════════════════════════════════

def ejecutar_escenario(hechos_activar: List[str]) -> Dict[str, Any]:
    """
    Función pública reutilizable por Streamlit y por los agentes.
    Crea un motor nuevo, asierta los hechos indicados y devuelve un dict
    estructurado con conclusiones y traza.
    """
    motor = MotorExperto()
    for nombre in hechos_activar:
        motor.asertar_hecho(nombre, confianza=0.95, origen="escenario")
    conclusiones = motor.inferir()

    return {
        "hechos_iniciales": hechos_activar,
        "hechos_activos"  : [h.nombre for h in motor.hechos_activos()],
        "conclusiones"    : [{
            "regla_id"    : c.regla_id,
            "hecho_nuevo" : c.hecho_nuevo,
            "accion"      : c.accion,
            "prioridad"   : c.prioridad,
            "confianza"   : round(c.confianza, 3),
            "antecedentes": c.antecedentes,
        } for c in conclusiones],
        "traza"           : motor.traza,
        "n_conclusiones"  : len(conclusiones),
        "prioridad_max"   : max([c.prioridad for c in conclusiones], default=0),
    }


def main() -> dict:
    print(f"\n{BOLD}{'='*70}")
    print("  PARTES D + E — SISTEMA EXPERTO Y REPRESENTACIÓN DEL CONOCIMIENTO")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{RESET}")

    # 1) Mostrar la base de conocimiento (Parte E)
    imprimir_base_conocimiento()
    imprimir_tabla_conocimiento()

    # 2) Ejecutar todos los escenarios (Parte D)
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  EJECUCIÓN DE ESCENARIOS — MOTOR DE INFERENCIA HACIA ADELANTE")
    print(f"{'═'*70}{RESET}")

    reportes: Dict[str, Any] = {}
    for nombre, hechos in ESCENARIOS_EXPERTO.items():
        motor = MotorExperto()
        for h in hechos:
            motor.asertar_hecho(h, confianza=0.95, origen="escenario")
        motor.inferir()
        imprimir_resultado_escenario(nombre, motor)

        reportes[nombre] = {
            "hechos_asertados": hechos,
            "conclusiones": [{
                "regla_id"   : c.regla_id,
                "hecho_nuevo": c.hecho_nuevo,
                "accion"     : c.accion,
                "prioridad"  : c.prioridad,
                "confianza"  : round(c.confianza, 3),
            } for c in motor.conclusiones],
        }

    # 3) Resumen
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  RESUMEN — PARTES D y E")
    print(f"{'═'*70}{RESET}")
    print(f"  Hechos definidos                : {len(HECHOS_BASE)}")
    print(f"  Reglas definidas                : {len(REGLAS)}")
    print(f"  Relaciones causales mapeadas    : {len(RELACIONES_CAUSALES)}")
    print(f"  Escenarios ejecutados           : {len(ESCENARIOS_EXPERTO)}")

    # 4) Guardar reporte
    from paths import REPORTE_SISTEMA_EXPERTO, asegurar_carpetas
    asegurar_carpetas()
    ruta = REPORTE_SISTEMA_EXPERTO

    reporte_final = {
        "modulo"             : "Partes D + E — Sistema experto (sistema_experto.py)",
        "timestamp"          : datetime.now().isoformat(),
        "hechos_base"        : [{
            "nombre"     : h.nombre,
            "descripcion": h.descripcion,
            "dominio"    : h.dominio,
        } for h in HECHOS_BASE.values()],
        "reglas"             : [{
            "id"          : r.id,
            "descripcion" : r.descripcion,
            "antecedentes": r.antecedentes,
            "consecuente" : r.consecuente,
            "accion"      : r.accion,
            "prioridad"   : r.prioridad,
        } for r in REGLAS],
        "tabla_conocimiento" : construir_tabla_conocimiento(),
        "relaciones_causales": RELACIONES_CAUSALES,
        "escenarios"         : reportes,
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte_final, f, ensure_ascii=False, indent=2)

    print(f"\n  {GREEN}Reporte JSON guardado: {ruta}{RESET}")
    print(f"\n{BOLD}{'='*70}")
    print("  Partes D y E completadas exitosamente.")
    print(f"{'='*70}{RESET}\n")
    return reporte_final


if __name__ == "__main__":
    main()
