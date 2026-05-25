"""
===============================================================================
PARTE F — AGENTES BASADOS EN CONOCIMIENTO
Sistema Inteligente Multiagente para Almacén Logístico
Autor: Persona 3 (Arquitecto de la inteligencia del sistema)
===============================================================================

Este módulo define el esqueleto lógico de los 5 agentes inteligentes que
componen el sistema. Cada agente es una CLASE de Python con:

   1. CONOCIMIENTO       — qué sabe y qué reglas/datos maneja.
   2. ENTRADA            — qué mensajes/datos recibe.
   3. SALIDA             — qué produce y a quién se lo entrega.
   4. CRITERIO DE DECISIÓN — cómo procesa la entrada para generar la salida.

Diseño compatible con la Parte J (concurrencia con threading + queue.Queue)
que implementa el integrante 4 (Beckam). Cada agente expone:

      agente.procesar(mensaje)   →  devuelve un Mensaje (o None)
      agente.estado              →  dict con métricas de ejecución

Esto permite que cada agente corra en su propio hilo leyendo de una cola
de entrada y publicando en una cola de salida, sin que su lógica interna
necesite ser modificada.
===============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import json
import random
import sys

# Forzar UTF-8 en la salida (necesario para Windows / cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Importaciones internas del proyecto
import sistema_experto as se

# ── Importación opcional de logica_probabilidad (Parte H) ─────────────────────
# Si el módulo está disponible, el AgenteCoordinador lo usará para calcular
# el riesgo global probabilístico. Si no, se utiliza una versión simplificada.
try:
    import logica_probabilidad as lp
    PROB_DISPONIBLE = True
except Exception:
    PROB_DISPONIBLE = False

# ── Colores consola ────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROTOCOLO DE COMUNICACIÓN — Mensaje (Parte I — apoyo a Beckam)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Todos los agentes intercambian objetos Mensaje. Esto permite que el
# Agente Coordinador y los hilos concurrentes sepan exactamente quién envió
# qué y por qué. Es el protocolo común del bus de mensajes.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Mensaje:
    """Unidad básica de comunicación entre agentes."""
    origen      : str                # Agente emisor
    destino     : str                # Agente destinatario ("BROADCAST" para todos)
    tipo        : str                # lectura | analisis | alerta | decision | comando
    contenido   : Dict[str, Any]     # Payload (datos, hechos, conclusiones…)
    prioridad   : int = 1            # 1=baja, 2=media, 3=alta, 4=crítica
    timestamp   : str = field(default_factory=lambda: datetime.now().isoformat())
    id_mensaje  : str = field(default_factory=lambda: f"M-{random.randint(1000,9999)}")

    def resumen(self) -> str:
        col = RED if self.prioridad >= 3 else (YELLOW if self.prioridad == 2 else GREEN)
        return (f"{col}[{self.tipo.upper():<9}]{RESET} "
                f"{DIM}{self.origen}{RESET} → {DIM}{self.destino}{RESET}  "
                f"prio={self.prioridad}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLASE BASE — Agente
# ═══════════════════════════════════════════════════════════════════════════════

class Agente:
    """
    Clase base abstracta para todos los agentes del sistema.

    Define la interfaz común que el módulo de concurrencia (Parte J)
    consumirá. Cada agente concreto debe sobreescribir:
        - conocimiento (qué sabe)
        - procesar(mensaje)  (cómo decide)
    """

    def __init__(self, nombre: str):
        self.nombre        = nombre
        self.conocimiento  : Dict[str, Any] = {}
        self.entradas_proc : int = 0
        self.salidas_emit  : int = 0
        self.log           : List[str] = []

    # ── Interfaz pública que los hilos llamarán ──────────────────────────────
    def procesar(self, mensaje: Optional[Mensaje]) -> Optional[Mensaje]:
        """
        Procesa un mensaje de entrada y devuelve un mensaje de salida (o None).
        Implementación por defecto: solo registra y retransmite.
        """
        raise NotImplementedError("Cada agente debe implementar procesar()")

    # ── Utilidades de logging ────────────────────────────────────────────────
    def _log(self, txt: str) -> None:
        marca = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log.append(f"[{marca}] {self.nombre}: {txt}")

    @property
    def estado(self) -> Dict[str, Any]:
        return {
            "agente"      : self.nombre,
            "entradas"    : self.entradas_proc,
            "salidas"     : self.salidas_emit,
            "conocimiento": list(self.conocimiento.keys()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AGENTES CONCRETOS (5 agentes — cumple el mínimo de 4 de la Parte F)
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# AGENTE 1 — SENSOR
# ──────────────────────────────────────────────────────────────────────────────
#  CONOCIMIENTO  : ubicación de los sensores, frecuencia de muestreo,
#                  umbrales nominales de cada variable.
#  ENTRADA       : (ninguna; lee del CSV o genera lecturas simuladas)
#  SALIDA        : Mensaje(tipo="lectura") con dict de variables ambientales.
#  DECISIÓN      : muestrea cada N segundos y publica una lectura cruda.
# ──────────────────────────────────────────────────────────────────────────────
class AgenteSensor(Agente):
    """Agente que produce o lee las lecturas de los sensores físicos."""

    def __init__(self, archivo_csv: Optional[str] = None):
        super().__init__("AgenteSensor")
        self.conocimiento = {
            "sensores_disponibles": ["temperatura", "humedad", "vibracion", "ocupacion"],
            "umbrales_nominales"  : {"temperatura": 30, "humedad": 75,
                                      "vibracion": 20, "ocupacion": 90},
            "fuente"              : archivo_csv or "simulacion_en_vivo",
        }
        self._datos_csv: List[Dict[str, float]] = []
        self._cursor   = 0

        if archivo_csv and Path(archivo_csv).exists():
            self._cargar_csv(archivo_csv)

    def _cargar_csv(self, ruta: str) -> None:
        """Carga el CSV generado por la Parte A (Gaby)."""
        try:
            import pandas as pd
            df = pd.read_csv(ruta)
            self._datos_csv = df.to_dict(orient="records")
            self._log(f"CSV cargado: {len(self._datos_csv)} registros")
        except ImportError:
            # Fallback sin pandas: lectura nativa CSV
            import csv
            with open(ruta, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._datos_csv = [
                    {k: float(v) if k != "timestamp" else v for k, v in row.items()}
                    for row in reader
                ]

    def procesar(self, mensaje: Optional[Mensaje] = None) -> Optional[Mensaje]:
        """Produce una nueva lectura (no consume; es el productor del pipeline)."""
        self.entradas_proc += 1

        if self._datos_csv:
            lectura = self._datos_csv[self._cursor % len(self._datos_csv)]
            self._cursor += 1
        else:
            # Lectura simulada
            lectura = {
                "timestamp"  : datetime.now().isoformat(),
                "temperatura": round(random.gauss(22, 3), 2),
                "humedad"    : round(random.gauss(55, 8), 2),
                "vibracion"  : round(random.gauss(11, 2), 2),
                "ocupacion"  : random.randint(40, 95),
            }

        self.salidas_emit += 1
        self._log(f"lectura emitida → T={lectura.get('temperatura',0):.1f}°C, "
                  f"V={lectura.get('vibracion',0):.1f}Hz")

        return Mensaje(
            origen   = self.nombre,
            destino  = "AgenteAnalizadorSenales",
            tipo     = "lectura",
            contenido= lectura,
            prioridad= 1,
        )


# ──────────────────────────────────────────────────────────────────────────────
# AGENTE 2 — ANALIZADOR DE SEÑALES
# ──────────────────────────────────────────────────────────────────────────────
#  CONOCIMIENTO  : umbrales de anomalía, ventana temporal, función de
#                  conversión lectura → hechos lógicos (Parte D).
#  ENTRADA       : Mensaje(tipo="lectura") proveniente del AgenteSensor.
#  SALIDA        : Mensaje(tipo="analisis") con lista de hechos asertados.
#  DECISIÓN      : compara cada variable con su umbral; si excede, deriva
#                  el hecho lógico correspondiente con su confianza.
# ──────────────────────────────────────────────────────────────────────────────
class AgenteAnalizadorSenales(Agente):
    """Agente que interpreta lecturas de sensores y deriva hechos lógicos."""

    def __init__(self, umbrales: Optional[Dict[str, float]] = None):
        super().__init__("AgenteAnalizadorSenales")
        self.conocimiento = {
            "umbrales"             : umbrales or {"temperatura": 30, "humedad": 75,
                                                    "vibracion": 20, "ocupacion": 90},
            "variables_supervisadas": ["temperatura", "humedad", "vibracion", "ocupacion"],
            "metodo_inferencia"     : "umbralización + conversión simbólica",
        }
        self._historial: List[Dict[str, float]] = []  # Ventana móvil

    def procesar(self, mensaje: Optional[Mensaje]) -> Optional[Mensaje]:
        if mensaje is None or mensaje.tipo != "lectura":
            return None
        self.entradas_proc += 1
        lectura = mensaje.contenido
        self._historial.append(lectura)
        if len(self._historial) > 60:
            self._historial.pop(0)

        hechos = se.hechos_desde_sensores(lectura, self.conocimiento["umbrales"])

        # Detección de sensor fallido: si la lectura es 0 o None en alguna variable
        for var in self.conocimiento["variables_supervisadas"]:
            if lectura.get(var) in (None, 0) and len(self._historial) > 2:
                hechos.append(("sensor_fallido", 0.70, f"sensor_caido:{var}"))
                break

        prioridad = 1
        if any(n in ("vibracion_anomala", "temperatura_alta", "humedad_excesiva")
                for n, _, _ in hechos):
            prioridad = 3

        self.salidas_emit += 1
        self._log(f"derivó {len(hechos)} hechos: "
                  f"{[h[0] for h in hechos] or 'ninguno'}")

        return Mensaje(
            origen   = self.nombre,
            destino  = "AgenteDecisor",
            tipo     = "analisis",
            contenido= {"hechos": hechos, "lectura": lectura},
            prioridad= prioridad,
        )


# ──────────────────────────────────────────────────────────────────────────────
# AGENTE 3 — ANALIZADOR DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────
#  CONOCIMIENTO  : reporte JSON del módulo vision_cv (Parte C — Cano).
#  ENTRADA       : Mensaje(tipo="comando", contenido={"accion":"analizar_imagenes"})
#                  o evento periódico.
#  SALIDA        : Mensaje(tipo="analisis") con hechos derivados de visión.
#  DECISIÓN      : lee el último reporte de visión y traduce sus diagnósticos
#                  a hechos lógicos (paquete_danado, ruta_obstruida).
# ──────────────────────────────────────────────────────────────────────────────
class AgenteAnalizadorImagenes(Agente):
    """Agente que extrae hechos lógicos del módulo de visión (Parte C)."""

    def __init__(self, ruta_reporte_vision: Optional[str] = None):
        super().__init__("AgenteAnalizadorImagenes")
        base = Path(__file__).parent / "images_output"
        self.conocimiento = {
            "ruta_reporte"      : ruta_reporte_vision or str(base / "reporte_vision.json"),
            "tecnicas_conocidas": ["Canny", "Otsu", "Umbral adaptativo", "HSV"],
            "diagnosticos"      : ["paquete_danado", "ruta_obstruida", "estado_ok"],
        }

    def _leer_reporte(self) -> Optional[Dict[str, Any]]:
        ruta = Path(self.conocimiento["ruta_reporte"])
        if not ruta.exists():
            return None
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self._log(f"error leyendo reporte: {e}")
            return None

    def procesar(self, mensaje: Optional[Mensaje] = None) -> Optional[Mensaje]:
        self.entradas_proc += 1
        reporte = self._leer_reporte()

        if reporte is None:
            # Fallback: simular un reporte si vision_cv no se ha ejecutado aún
            self._log("sin reporte_vision.json — usando reporte simulado")
            reporte = {
                "resultados": [
                    {"imagen": "img_danado.jpg",   "danio_detectado": True,
                     "nivel_alerta": "ALTO"},
                    {"imagen": "img_obstruida.jpg","obstruccion_detectada": True,
                     "nivel_alerta": "CRITICO"},
                ]
            }

        hechos = se.hechos_desde_vision(reporte)
        prioridad = 3 if hechos else 1

        self.salidas_emit += 1
        self._log(f"derivó {len(hechos)} hechos de visión: "
                  f"{[h[0] for h in hechos] or 'ninguno'}")

        return Mensaje(
            origen   = self.nombre,
            destino  = "AgenteDecisor",
            tipo     = "analisis",
            contenido= {"hechos": hechos, "fuente": "vision_cv"},
            prioridad= prioridad,
        )


# ──────────────────────────────────────────────────────────────────────────────
# AGENTE 4 — DECISOR
# ──────────────────────────────────────────────────────────────────────────────
#  CONOCIMIENTO  : base de hechos y reglas del sistema experto (Parte D).
#  ENTRADA       : Mensajes(tipo="analisis") provenientes de los analizadores.
#  SALIDA        : Mensaje(tipo="decision") con conclusiones priorizadas.
#  DECISIÓN      : asierta los hechos recibidos en su motor experto y ejecuta
#                  forward chaining para derivar conclusiones y acciones.
# ──────────────────────────────────────────────────────────────────────────────
class AgenteDecisor(Agente):
    """Agente que aplica las reglas del sistema experto y emite decisiones."""

    def __init__(self):
        super().__init__("AgenteDecisor")
        self.motor = se.MotorExperto()
        self.conocimiento = {
            "n_hechos_base"   : len(se.HECHOS_BASE),
            "n_reglas"        : len(se.REGLAS),
            "metodo"          : "Forward chaining",
            "ultima_decision" : None,
        }

    def procesar(self, mensaje: Optional[Mensaje]) -> Optional[Mensaje]:
        if mensaje is None or mensaje.tipo != "analisis":
            return None
        self.entradas_proc += 1

        # Resetear hechos derivados manteniendo la base
        self.motor = se.MotorExperto()

        for nombre, confianza, origen in mensaje.contenido.get("hechos", []):
            self.motor.asertar_hecho(nombre, confianza=confianza, origen=origen)

        conclusiones = self.motor.inferir()
        decisiones = [{
            "regla"     : c.regla_id,
            "conclusion": c.hecho_nuevo,
            "accion"    : c.accion,
            "prioridad" : c.prioridad,
            "confianza" : round(c.confianza, 3),
        } for c in conclusiones]

        self.conocimiento["ultima_decision"] = decisiones
        prioridad = max([c.prioridad for c in conclusiones], default=1)

        self.salidas_emit += 1
        self._log(f"infirió {len(decisiones)} conclusiones — "
                  f"prioridad máx: {prioridad}")

        return Mensaje(
            origen   = self.nombre,
            destino  = "AgenteCoordinador",
            tipo     = "decision",
            contenido= {
                "decisiones"      : decisiones,
                "hechos_iniciales": [n for n, _, _ in mensaje.contenido.get("hechos", [])],
                "n_alertas"       : len([d for d in decisiones if d["prioridad"] >= 3]),
            },
            prioridad= prioridad,
        )


# ──────────────────────────────────────────────────────────────────────────────
# AGENTE 5 — COORDINADOR
# ──────────────────────────────────────────────────────────────────────────────
#  CONOCIMIENTO  : política global de operación, mapa de prioridades, ruta
#                  hacia el motor probabilístico (Parte H).
#  ENTRADA       : Mensajes(tipo="decision") de uno o varios Decisores.
#  SALIDA        : Mensaje(tipo="alerta") con el plan global de acciones y
#                  el nivel de riesgo operativo agregado.
#  DECISIÓN      : fusiona las decisiones, calcula el riesgo global vía
#                  Noisy-OR (Parte H) y emite la directiva final al dashboard.
# ──────────────────────────────────────────────────────────────────────────────
class AgenteCoordinador(Agente):
    """Agente que integra las decisiones de todos los demás agentes."""

    def __init__(self):
        super().__init__("AgenteCoordinador")
        self.conocimiento = {
            "politica"           : "maximizar seguridad > continuidad > velocidad",
            "fuente_probabilidad": "logica_probabilidad.py" if PROB_DISPONIBLE else "fallback",
            "decisiones_recibidas": [],
        }
        self._motor_prob = lp.MotorProbabilistico() if PROB_DISPONIBLE else None

    def procesar(self, mensaje: Optional[Mensaje]) -> Optional[Mensaje]:
        if mensaje is None or mensaje.tipo != "decision":
            return None
        self.entradas_proc += 1
        contenido = mensaje.contenido
        decisiones = contenido.get("decisiones", [])
        self.conocimiento["decisiones_recibidas"].append(decisiones)

        # Riesgo global: Noisy-OR sobre las confianzas (si está disponible Parte H)
        confianzas = [d["confianza"] for d in decisiones if d["prioridad"] >= 2]
        if self._motor_prob and confianzas:
            p_global = self._motor_prob.fusionar_evidencias(confianzas)
            nivel, _ = self._motor_prob.nivel_alerta(p_global)
        else:
            # Fallback simple: máximo
            p_global = max(confianzas) if confianzas else 0.0
            nivel = ("CRÍTICO" if p_global > 0.75 else
                     "ALTO"    if p_global > 0.55 else
                     "MEDIO"   if p_global > 0.30 else
                     "BAJO")

        plan = self._construir_plan(decisiones, nivel, p_global)
        self.salidas_emit += 1
        self._log(f"emitió plan global — nivel: {nivel} ({p_global:.0%})")

        return Mensaje(
            origen   = self.nombre,
            destino  = "BROADCAST",
            tipo     = "alerta",
            contenido= plan,
            prioridad= 4 if nivel == "CRÍTICO" else (3 if nivel == "ALTO" else 2),
        )

    def _construir_plan(self, decisiones: List[Dict[str, Any]],
                          nivel: str, p_global: float) -> Dict[str, Any]:
        """Crea el plan de acción global ordenado por prioridad."""
        decisiones_ordenadas = sorted(
            decisiones, key=lambda d: (-d["prioridad"], -d["confianza"])
        )
        if nivel == "CRÍTICO":
            directiva = "DETENER OPERACIONES — protocolo de emergencia."
        elif nivel == "ALTO":
            directiva = "REDUCIR RITMO — notificar supervisor y técnico."
        elif nivel == "MEDIO":
            directiva = "OPERAR CON PRECAUCIÓN — incrementar monitoreo."
        else:
            directiva = "OPERACIÓN NORMAL — sin intervención requerida."

        return {
            "nivel_global"     : nivel,
            "riesgo_global_pct": round(p_global * 100, 2),
            "directiva_principal": directiva,
            "acciones_priorizadas": [{
                "orden"    : i + 1,
                "prioridad": d["prioridad"],
                "accion"   : d["accion"],
                "regla"    : d["regla"],
                "conclusion": d["conclusion"],
            } for i, d in enumerate(decisiones_ordenadas)],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SISTEMA MULTIAGENTE — Orquestador (modo secuencial, sin threads)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Esta clase permite ejecutar el flujo completo sensor → analizadores →
# decisor → coordinador SIN concurrencia para fines de demostración y
# pruebas unitarias. La Parte J (Beckam) sustituirá este loop por hilos.
# ═══════════════════════════════════════════════════════════════════════════════

class SistemaMultiAgente:
    """Orquestador secuencial del pipeline multiagente."""

    def __init__(self, csv_sensores: Optional[str] = None,
                 reporte_vision: Optional[str] = None):
        self.sensor      = AgenteSensor(csv_sensores)
        self.an_senales  = AgenteAnalizadorSenales()
        self.an_imagenes = AgenteAnalizadorImagenes(reporte_vision)
        self.decisor     = AgenteDecisor()
        self.coordinador = AgenteCoordinador()
        self.historial   : List[Mensaje] = []

    def agentes(self) -> List[Agente]:
        return [self.sensor, self.an_senales, self.an_imagenes,
                self.decisor, self.coordinador]

    def ciclo(self, incluir_vision: bool = True) -> Dict[str, Any]:
        """
        Ejecuta un ciclo completo del pipeline y devuelve el plan global
        producido por el coordinador.
        """
        # 1) Sensor → AnalizadorSenales
        m_sensor = self.sensor.procesar(None)
        self.historial.append(m_sensor)
        m_senales = self.an_senales.procesar(m_sensor)
        self.historial.append(m_senales)

        # 2) AnalizadorImagenes (paralelo conceptual; aquí va después)
        m_imagenes = None
        if incluir_vision:
            m_imagenes = self.an_imagenes.procesar(None)
            self.historial.append(m_imagenes)

        # 3) Decisor: fusiona ambos análisis
        hechos_fusionados: List[tuple] = []
        for m in (m_senales, m_imagenes):
            if m is not None:
                hechos_fusionados.extend(m.contenido.get("hechos", []))

        m_analisis_fusion = Mensaje(
            origen   = "FUSION",
            destino  = "AgenteDecisor",
            tipo     = "analisis",
            contenido= {"hechos": hechos_fusionados},
            prioridad= 2,
        )
        m_decision = self.decisor.procesar(m_analisis_fusion)
        self.historial.append(m_decision)

        # 4) Coordinador → directiva final
        m_alerta = self.coordinador.procesar(m_decision)
        self.historial.append(m_alerta)

        return {
            "mensajes_intercambiados": len(self.historial),
            "plan_global"            : m_alerta.contenido if m_alerta else {},
            "prioridad_global"       : m_alerta.prioridad if m_alerta else 1,
        }

    def resumen_estado(self) -> List[Dict[str, Any]]:
        return [a.estado for a in self.agentes()]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PRESENTACIÓN (consola)
# ═══════════════════════════════════════════════════════════════════════════════

def imprimir_descripcion_agentes():
    """Tabla descriptiva de los 4 agentes — evidencia Parte F."""
    info = [
        ("AgenteSensor",
         "umbrales, sensores",
         "—",
         "Mensaje 'lectura' con vars ambientales",
         "muestrea cada ciclo del CSV o simula lectura"),
        ("AgenteAnalizadorSenales",
         "umbrales por variable + ventana móvil",
         "Mensaje 'lectura'",
         "Mensaje 'analisis' con hechos lógicos",
         "compara variables vs umbrales y deriva hechos"),
        ("AgenteAnalizadorImagenes",
         "reporte JSON de vision_cv",
         "Mensaje 'comando' o evento periódico",
         "Mensaje 'analisis' con hechos de visión",
         "traduce niveles de alerta visual a hechos"),
        ("AgenteDecisor",
         "8 hechos + 8 reglas del sist. experto",
         "Mensaje 'analisis' de los analizadores",
         "Mensaje 'decision' con conclusiones",
         "aplica forward chaining sobre las reglas"),
        ("AgenteCoordinador",
         "política global + motor probabilístico",
         "Mensajes 'decision' de los decisores",
         "Mensaje 'alerta' con plan global",
         "fusiona por Noisy-OR y emite directiva"),
    ]

    print(f"\n{CYAN}{BOLD}{'─'*70}")
    print("  DESCRIPCIÓN DE LOS AGENTES (Parte F)")
    print(f"{'─'*70}{RESET}\n")

    for nombre, conoce, entrada, salida, decide in info:
        print(f"  {BOLD}{MAGENTA}► {nombre}{RESET}")
        print(f"    {DIM}Conocimiento:{RESET}  {conoce}")
        print(f"    {DIM}Entrada    :{RESET}  {entrada}")
        print(f"    {DIM}Salida     :{RESET}  {salida}")
        print(f"    {DIM}Cómo decide:{RESET}  {decide}\n")


def imprimir_resultado_ciclo(resultado: Dict[str, Any]):
    plan = resultado.get("plan_global", {})
    nivel = plan.get("nivel_global", "BAJO")
    color = (RED if nivel == "CRÍTICO" else
             YELLOW if nivel == "ALTO" else
             GREEN if nivel == "BAJO" else CYAN)

    print(f"\n  {BOLD}Plan global emitido por el Coordinador:{RESET}")
    print(f"    Nivel de riesgo : {color}{nivel}{RESET}  "
          f"({plan.get('riesgo_global_pct', 0)}%)")
    print(f"    Directiva       : {color}{plan.get('directiva_principal','—')}{RESET}")

    acciones = plan.get("acciones_priorizadas", [])
    if acciones:
        print(f"    Acciones priorizadas:")
        for a in acciones[:5]:
            col_p = RED if a["prioridad"] == 4 else (YELLOW if a["prioridad"] == 3 else GREEN)
            print(f"      {col_p}[P{a['prioridad']}]{RESET} {a['accion']}  "
                  f"{DIM}({a['regla']} → {a['conclusion']}){RESET}")
    else:
        print(f"    {GREEN}Sin acciones requeridas — operación normal.{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    print(f"\n{BOLD}{'='*70}")
    print("  PARTE F — AGENTES BASADOS EN CONOCIMIENTO")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{RESET}")

    # 1) Descripción formal de los agentes
    imprimir_descripcion_agentes()

    # 2) Ejecución de un sistema multiagente con datos reales
    csv_path = Path(__file__).resolve().parent.parent / "data" / "simulacion_almacen.csv"
    sistema = SistemaMultiAgente(
        csv_sensores=str(csv_path) if csv_path.exists() else None
    )

    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  EJECUCIÓN — 3 ciclos del pipeline multiagente (modo secuencial)")
    print(f"{'═'*70}{RESET}")

    historial_ciclos: List[Dict[str, Any]] = []
    for i in range(1, 4):
        print(f"\n{YELLOW}{BOLD}  ▶ CICLO #{i}{RESET}")
        print(f"  {'─'*65}")
        resultado = sistema.ciclo(incluir_vision=(i == 1))
        imprimir_resultado_ciclo(resultado)
        historial_ciclos.append(resultado)

    # 3) Resumen de estados
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  ESTADO FINAL DE LOS AGENTES")
    print(f"{'═'*70}{RESET}")
    print(f"  {'AGENTE':<28} {'ENTRADAS':>9} {'SALIDAS':>9} CONOCIMIENTO")
    print(f"  {'─'*26}  {'─'*8}  {'─'*7}  {'─'*16}")
    for est in sistema.resumen_estado():
        print(f"  {BOLD}{est['agente']:<28}{RESET} {est['entradas']:>9} "
              f"{est['salidas']:>9}  {DIM}{len(est['conocimiento'])} ítems{RESET}")

    # 4) Guardar reporte JSON
    salida = Path(__file__).parent / "images_output"
    salida.mkdir(exist_ok=True)
    ruta = salida / "reporte_agentes.json"

    reporte = {
        "modulo"          : "Parte F — Agentes (agentes.py)",
        "timestamp"       : datetime.now().isoformat(),
        "agentes_definidos": [
            {"nombre": a.nombre, "conocimiento": list(a.conocimiento.keys())}
            for a in sistema.agentes()
        ],
        "ciclos_ejecutados": len(historial_ciclos),
        "historial_ciclos" : historial_ciclos,
        "mensajes_totales" : len(sistema.historial),
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print(f"\n  {GREEN}Reporte JSON guardado: {ruta}{RESET}")
    print(f"\n{BOLD}{'='*70}")
    print("  Parte F completada exitosamente.")
    print(f"{'='*70}{RESET}\n")
    return reporte


if __name__ == "__main__":
    main()
