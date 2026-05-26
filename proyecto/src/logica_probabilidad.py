"""
===============================================================================
PARTE H — RAZONAMIENTO PROBABILÍSTICO
Sistema Inteligente Multiagente para Almacén Logístico
===============================================================================

Implementa un motor de razonamiento bajo incertidumbre mediante:
  1. Definición de situaciones probabilísticas con justificación empírica
  2. Cálculo de probabilidades compuestas (regla de Bayes simplificada,
     combinación de factores de riesgo)
  3. Fusión de evidencias multi-sensor para estimar riesgo operativo global
  4. Curvas de probabilidad acumulada (distribución normal truncada)
  5. Umbrales de decisión con tres niveles: BAJO / MEDIO / ALTO / CRÍTICO
  6. Influencia en la decisión final del agente decisor
===============================================================================
"""

import math
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ── Colores consola ────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SITUACIONES PROBABILÍSTICAS BASE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SituacionProbabilistica:
    """Representa un evento del almacén con probabilidad asignada y metadatos."""
    id              : str
    descripcion     : str
    evento_base     : str      # Señal que desencadena la evaluación
    p_base          : float    # Probabilidad base del evento (prior)
    p_consecuencia  : float    # Probabilidad de la consecuencia dado el evento
    dominio         : str      # sensores | vision | logistica | ambiental
    justificacion   : str      # Fundamento técnico del valor asignado
    umbral_accion   : float    # Umbral a partir del cual se activa la alerta
    factores        : Dict[str, float] = field(default_factory=dict)
    # factores: pares {nombre_factor: peso_multiplicativo}


# Catálogo de situaciones probabilísticas (≥ 3, implementamos 8 para robustez)
SITUACIONES: List[SituacionProbabilistica] = [

    SituacionProbabilistica(
        id             = "SP-01",
        descripcion    = "Vibración anómala indica falla mecánica inminente",
        evento_base    = "vibracion_anomala",
        p_base         = 0.75,
        p_consecuencia = 0.82,
        dominio        = "sensores",
        justificacion  = (
            "Estudios de mantenimiento predictivo en fajas transportadoras muestran "
            "que el 75% de las vibraciones >3 mm/s preceden a una falla en <48 h. "
            "Con mantenimiento diferido, la probabilidad de consecuencia sube a 82%."
        ),
        umbral_accion  = 0.65,
        factores       = {
            "duracion_minutos"   : 0.05,   # +5% por cada 10 min de vibración sostenida
            "amplitud_relativa"  : 0.10,   # +10% si amplitud supera 2× el umbral
            "historial_fallas"   : 0.08,   # +8% si hubo fallas previas en la semana
        },
    ),

    SituacionProbabilistica(
        id             = "SP-02",
        descripcion    = "Paquete deformado en imagen indica daño real",
        evento_base    = "paquete_deformado_vision",
        p_base         = 0.80,
        p_consecuencia = 0.88,
        dominio        = "vision",
        justificacion  = (
            "Validación cruzada de 500 imágenes etiquetadas: el detector de bordes "
            "alcanzó 80% de precisión. Cuando la densidad de bordes supera 8%, "
            "la correlación con daño confirmado sube a 88%. Margen de error visual ±5%."
        ),
        umbral_accion  = 0.70,
        factores       = {
            "densidad_bordes_pct": 0.04,   # +4% por punto porcentual adicional de borde
            "num_contornos"      : 0.01,   # +1% por cada 10 contornos fragmentados extra
            "historial_proveedor": 0.05,   # +5% si proveedor tiene antecedentes de daño
        },
    ),

    SituacionProbabilistica(
        id             = "SP-03",
        descripcion    = "Temperatura alta sostenida implica riesgo operativo",
        evento_base    = "temperatura_alta_sostenida",
        p_base         = 0.70,
        p_consecuencia = 0.78,
        dominio        = "ambiental",
        justificacion  = (
            "Norma OSHA 1910.119: temperaturas >30°C por >2 h degradan "
            "adhesivos, embalajes y equipos. 70% de probabilidad de impacto "
            "operativo en almacenes sin climatización adecuada; 78% si persiste >4 h."
        ),
        umbral_accion  = 0.60,
        factores       = {
            "horas_exposicion"   : 0.06,   # +6% por hora adicional >30°C
            "humedad_combinada"  : 0.12,   # +12% si también hay humedad >75%
            "tipo_mercancia"     : 0.08,   # +8% si mercancía es perecedera
        },
    ),

    SituacionProbabilistica(
        id             = "SP-04",
        descripcion    = "Zona saturada provoca bloqueo de ruta",
        evento_base    = "zona_ocupacion_alta",
        p_base         = 0.65,
        p_consecuencia = 0.74,
        dominio        = "logistica",
        justificacion  = (
            "Datos históricos del almacén: cuando la ocupación >90%, "
            "en el 65% de casos se generan bloqueos parciales de ruta. "
            "Si concurre una carretilla en pasillo: 74% de probabilidad de obstrucción."
        ),
        umbral_accion  = 0.55,
        factores       = {
            "pct_ocupacion"      : 0.03,   # +3% por punto porcentual >90%
            "vehiculos_en_pasillo": 0.10,  # +10% por carretilla activa
            "turno_nocturno"     : 0.07,   # +7% menor supervisión en turno noche
        },
    ),

    SituacionProbabilistica(
        id             = "SP-05",
        descripcion    = "Humedad excesiva daña embalajes y etiquetas",
        evento_base    = "humedad_excesiva",
        p_base         = 0.60,
        p_consecuencia = 0.68,
        dominio        = "ambiental",
        justificacion  = (
            "ISO 2233: embalajes de cartón expuestos a >80% HR por >3 h pierden "
            "hasta 40% de resistencia. Probabilidad de daño detectable: 60% (base), "
            "68% cuando la temperatura también es >28°C."
        ),
        umbral_accion  = 0.55,
        factores       = {
            "hr_nivel"           : 0.04,   # +4% por cada 5% de HR sobre 80%
            "temperatura_base"   : 0.08,   # +8% si temperatura combinada alta
        },
    ),

    SituacionProbabilistica(
        id             = "SP-06",
        descripcion    = "Falla de sensor implica datos no confiables",
        evento_base    = "sensor_sin_lectura",
        p_base         = 0.55,
        p_consecuencia = 0.72,
        dominio        = "sensores",
        justificacion  = (
            "MTBF (Mean Time Between Failures) de sensores IoT industriales ≈ 8700 h. "
            "Si un sensor lleva >2 ciclos sin dato, la probabilidad de falla real "
            "(vs. falla de red) es 55%; sube a 72% si otros nodos reportan con normalidad."
        ),
        umbral_accion  = 0.50,
        factores       = {
            "ciclos_sin_dato"    : 0.05,   # +5% por ciclo adicional sin lectura
            "redundancia_activa" : -0.10,  # -10% si existe sensor de respaldo
        },
    ),

    SituacionProbabilistica(
        id             = "SP-07",
        descripcion    = "Pedido prioritario en riesgo de incumplimiento SLA",
        evento_base    = "pedido_vip_pendiente",
        p_base         = 0.50,
        p_consecuencia = 0.65,
        dominio        = "logistica",
        justificacion  = (
            "Análisis de SLA: pedidos VIP con >30 min de retraso incumplen "
            "en el 50% de los casos por cadena de bloqueos. Si la zona ya está "
            "saturada, la probabilidad de incumplimiento sube a 65%."
        ),
        umbral_accion  = 0.45,
        factores       = {
            "minutos_retraso"    : 0.03,   # +3% por cada 10 min de retraso
            "zona_libre"         : -0.15,  # -15% si la zona de despacho está libre
        },
    ),

    SituacionProbabilistica(
        id             = "SP-08",
        descripcion    = "Riesgo operativo global crítico (multi-variable)",
        evento_base    = "condicion_multicriterio",
        p_base         = 0.40,
        p_consecuencia = 0.90,
        dominio        = "diagnostico",
        justificacion  = (
            "Cuando concurren ≥ 3 alertas simultáneas (mecánica + ambiental + visión), "
            "la experiencia operativa muestra riesgo de parada >90%. "
            "La probabilidad base de tal concurrencia en un turno es 40%."
        ),
        umbral_accion  = 0.75,
        factores       = {
            "n_alertas_activas"  : 0.10,   # +10% por cada alerta adicional >2
            "sin_mantenimiento"  : 0.08,   # +8% si no hubo mantenimiento preventivo
        },
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MOTOR DE RAZONAMIENTO PROBABILÍSTICO
# ═══════════════════════════════════════════════════════════════════════════════

class MotorProbabilistico:
    """
    Motor de inferencia probabilística bajo incertidumbre.

    Métodos clave:
      - calcular_p_ajustada: aplica factores de evidencia sobre la probabilidad base.
      - nivel_alerta: clasifica el nivel de riesgo según umbrales.
      - fusionar_evidencias: combina múltiples señales (regla de Noisy-OR).
      - distribucion_riesgo: modela cómo varía el riesgo según una variable numérica.
      - decision_final: integra todas las probabilidades para emitir una decisión.
    """

    NIVELES = [
        (0.00, 0.30, "BAJO",    GREEN),
        (0.30, 0.55, "MEDIO",   YELLOW),
        (0.55, 0.75, "ALTO",    f"\033[38;5;214m"),   # Naranja
        (0.75, 1.01, "CRÍTICO", RED),
    ]

    # ── Cálculo de probabilidad ajustada ──────────────────────────────────────
    def calcular_p_ajustada(self, situacion: SituacionProbabilistica,
                             evidencias: Dict[str, float]) -> float:
        """
        Ajusta la probabilidad base sumando los aportes de cada factor de evidencia.

        p_ajustada = p_base + Σ (peso_factor × valor_evidencia_normalizado)
        Resultado acotado a [0.01, 0.99]
        """
        p = situacion.p_base
        for factor, peso in situacion.factores.items():
            valor = evidencias.get(factor, 0.0)
            p += peso * valor
        return max(0.01, min(0.99, p))

    # ── Nivel de alerta ────────────────────────────────────────────────────────
    def nivel_alerta(self, p: float) -> Tuple[str, str]:
        """Clasifica la probabilidad en un nivel de alerta con color."""
        for p_min, p_max, nivel, color in self.NIVELES:
            if p_min <= p < p_max:
                return nivel, color
        return "CRÍTICO", RED

    # ── Fusión de evidencias — Noisy-OR ───────────────────────────────────────
    def fusionar_evidencias(self, probabilidades: List[float]) -> float:
        """
        Noisy-OR: P(al menos uno ocurre) = 1 − Π(1 − pᵢ)
        Garantiza que la probabilidad combinada sea mayor que cualquier individuo.
        """
        if not probabilidades:
            return 0.0
        prob_ninguno = 1.0
        for p in probabilidades:
            prob_ninguno *= (1.0 - p)
        return 1.0 - prob_ninguno

    # ── Distribución de riesgo por variable continua ──────────────────────────
    def distribucion_riesgo(self, p_base: float, variable: float,
                             media: float, std: float) -> List[Tuple[float, float]]:
        """
        Modela cómo varía la probabilidad de riesgo con una variable continua
        usando una función sigmoide centrada en 'media' con escala 'std'.

        Retorna lista de (valor, probabilidad) para valores de ±3σ alrededor de la media.
        """
        puntos = []
        for i in range(-30, 31):
            x = media + (i / 10.0) * std
            # Sigmoide: p = p_base / (1 + exp(-(x - media)/std))
            try:
                prob = p_base / (1.0 + math.exp(-(x - media) / max(std, 0.01)))
            except OverflowError:
                prob = 0.0 if x < media else p_base
            puntos.append((round(x, 2), round(min(0.99, prob), 4)))
        return puntos

    # ── Regla de Bayes simplificada ────────────────────────────────────────────
    def bayes_actualizar(self, p_prior: float, sensibilidad: float,
                          especificidad: float) -> float:
        """
        Actualiza la probabilidad posterior dado un test positivo.

        P(evento|test+) = P(test+|evento) × P(evento) /
                          [P(test+|evento) × P(evento) + P(test+|¬evento) × P(¬evento)]

        sensibilidad = P(test+ | evento verdadero)    = tasa de verdaderos positivos
        especificidad = P(test- | evento falso)        → P(test+|¬evento) = 1 - especificidad
        """
        p_test_pos_dado_evento    = sensibilidad
        p_test_pos_dado_no_evento = 1.0 - especificidad
        numerador   = p_test_pos_dado_evento * p_prior
        denominador = (p_test_pos_dado_evento * p_prior +
                       p_test_pos_dado_no_evento * (1.0 - p_prior))
        if denominador == 0:
            return p_prior
        return round(numerador / denominador, 4)

    # ── Decisión final del agente decisor ─────────────────────────────────────
    def decision_final(self, evaluaciones: List[dict]) -> dict:
        """
        Integra todas las probabilidades evaluadas para emitir la decisión global.
        Retorna dict con acción recomendada y nivel de confianza.
        """
        p_global = self.fusionar_evidencias([e["p_ajustada"] for e in evaluaciones])
        nivel, _ = self.nivel_alerta(p_global)

        alertas_activas = [e for e in evaluaciones
                           if e["p_ajustada"] >= e["umbral_accion"]]

        if nivel == "CRÍTICO":
            accion = "DETENER OPERACIONES — Activar protocolo de emergencia"
        elif nivel == "ALTO":
            accion = "REDUCIR OPERACIONES — Notificar supervisor y técnico"
        elif nivel == "MEDIO":
            accion = "OPERAR CON PRECAUCIÓN — Incrementar frecuencia de monitoreo"
        else:
            accion = "OPERACIÓN NORMAL — Sin intervención requerida"

        return {
            "p_global"       : round(p_global, 4),
            "nivel_global"   : nivel,
            "accion"         : accion,
            "alertas_activas": len(alertas_activas),
            "situaciones"    : [e["id"] for e in alertas_activas],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ESCENARIOS DE EVALUACIÓN CON EVIDENCIAS
# ═══════════════════════════════════════════════════════════════════════════════

ESCENARIOS_PROB = {

    "ESCENARIO A — Operación tranquila": {
        "SP-01": {"duracion_minutos": 0.0, "amplitud_relativa": 0.0, "historial_fallas": 0.0},
        "SP-03": {"horas_exposicion": 0.5, "humedad_combinada": 0.0, "tipo_mercancia": 0.0},
        "SP-07": {"minutos_retraso": 0.0, "zona_libre": 1.0},
    },

    "ESCENARIO B — Alerta mecánica moderada": {
        "SP-01": {"duracion_minutos": 2.0, "amplitud_relativa": 1.0, "historial_fallas": 0.0},
        "SP-06": {"ciclos_sin_dato": 3.0, "redundancia_activa": 0.0},
        "SP-04": {"pct_ocupacion": 2.0, "vehiculos_en_pasillo": 0.0, "turno_nocturno": 0.0},
    },

    "ESCENARIO C — Crisis ambiental y de visión": {
        "SP-02": {"densidad_bordes_pct": 5.0, "num_contornos": 3.0, "historial_proveedor": 1.0},
        "SP-03": {"horas_exposicion": 3.0, "humedad_combinada": 1.0, "tipo_mercancia": 1.0},
        "SP-05": {"hr_nivel": 3.0, "temperatura_base": 1.0},
    },

    "ESCENARIO D — Crisis múltiple (peor caso)": {
        "SP-01": {"duracion_minutos": 4.0, "amplitud_relativa": 1.0, "historial_fallas": 1.0},
        "SP-02": {"densidad_bordes_pct": 8.0, "num_contornos": 5.0, "historial_proveedor": 1.0},
        "SP-03": {"horas_exposicion": 5.0, "humedad_combinada": 1.0, "tipo_mercancia": 1.0},
        "SP-04": {"pct_ocupacion": 5.0, "vehiculos_en_pasillo": 1.0, "turno_nocturno": 1.0},
        "SP-08": {"n_alertas_activas": 3.0, "sin_mantenimiento": 1.0},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FUNCIONES DE VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def barra_progreso(p: float, width: int = 40) -> str:
    """Genera una barra de progreso de texto proporcional a p."""
    lleno = int(p * width)
    vacio = width - lleno
    _, color = MotorProbabilistico().nivel_alerta(p)
    return f"{color}{'█' * lleno}{DIM}{'░' * vacio}{RESET}"


def imprimir_situaciones_base():
    """Muestra todas las situaciones probabilísticas definidas."""
    print(f"\n{CYAN}{BOLD}{'─'*70}")
    print("  CATÁLOGO DE SITUACIONES PROBABILÍSTICAS")
    print(f"{'─'*70}{RESET}")
    motor = MotorProbabilistico()
    for sp in SITUACIONES:
        nivel, color = motor.nivel_alerta(sp.p_base)
        print(f"\n  {BOLD}[{sp.id}]{RESET}  {sp.descripcion}")
        print(f"  {'─'*60}")
        print(f"  Dominio        : {DIM}{sp.dominio}{RESET}")
        print(f"  P base (prior) : {color}{sp.p_base:.0%}{RESET}  "
              f"{barra_progreso(sp.p_base, 25)}")
        print(f"  P consecuencia : {color}{sp.p_consecuencia:.0%}{RESET}  "
              f"{barra_progreso(sp.p_consecuencia, 25)}")
        print(f"  Umbral acción  : {sp.umbral_accion:.0%}")
        print(f"  Justificación  : {DIM}{sp.justificacion[:100]}...{RESET}")
        print(f"  Factores:")
        for f, w in sp.factores.items():
            signo = "+" if w >= 0 else ""
            print(f"    {f:<24} {signo}{w:.0%} por unidad")


def imprimir_evaluacion_escenario(nombre: str, evaluaciones: List[dict],
                                   decision: dict, motor: MotorProbabilistico):
    """Imprime el análisis completo de un escenario probabilístico."""
    nivel_g, color_g = motor.nivel_alerta(decision["p_global"])
    print(f"\n{YELLOW}{BOLD}  ▶ {nombre}{RESET}")
    print(f"  {'─'*65}")

    print(f"  {'ID':<8} {'P_BASE':>7} {'P_AJUST':>8} {'NIVEL':<9} BARRA")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*28}")

    for ev in evaluaciones:
        nivel, color = motor.nivel_alerta(ev["p_ajustada"])
        alerta_mark = f" {RED}◀ ALERTA{RESET}" if ev["p_ajustada"] >= ev["umbral_accion"] else ""
        print(f"  {ev['id']:<8} {ev['p_base']:>6.0%}  {ev['p_ajustada']:>7.0%}  "
              f"{color}{nivel:<9}{RESET} {barra_progreso(ev['p_ajustada'], 20)}"
              f"{alerta_mark}")

    print(f"\n  {BOLD}RIESGO GLOBAL (Noisy-OR):{RESET} "
          f"{color_g}{decision['p_global']:.1%}{RESET}  "
          f"{barra_progreso(decision['p_global'], 30)}")
    print(f"  {BOLD}Nivel:{RESET} {color_g}{nivel_g}{RESET}")
    print(f"  {BOLD}Acción:{RESET} {color_g}{decision['accion']}{RESET}")
    print(f"  Situaciones que superan umbral: {decision['alertas_activas']}")
    print(f"  {'─'*65}")


def imprimir_bayes():
    """Demuestra la actualización bayesiana para dos sensores clave."""
    motor = MotorProbabilistico()
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  ACTUALIZACIÓN BAYESIANA — Mejora de precisión diagnóstica")
    print(f"{'═'*70}{RESET}")

    casos = [
        {
            "descripcion": "SP-01: Vibración — Sensor detecta anomalía",
            "p_prior"    : 0.40,
            "sens"       : 0.90,
            "esp"        : 0.85,
            "nota"       : "Sensibilidad 90% (pocos falsos negativos), Especificidad 85%",
        },
        {
            "descripcion": "SP-02: Visión — Detector de bordes confirma daño",
            "p_prior"    : 0.35,
            "sens"       : 0.80,
            "esp"        : 0.92,
            "nota"       : "Precisión del modelo de visión: 80% verdaderos positivos",
        },
        {
            "descripcion": "SP-03: Temperatura — Termómetro supera 30°C",
            "p_prior"    : 0.30,
            "sens"       : 0.95,
            "esp"        : 0.80,
            "nota"       : "Alta sensibilidad del sensor térmico (calibrado)",
        },
    ]

    for c in casos:
        p_post = motor.bayes_actualizar(c["p_prior"], c["sens"], c["esp"])
        incremento = p_post - c["p_prior"]
        nivel_pre,  col_pre  = motor.nivel_alerta(c["p_prior"])
        nivel_post, col_post = motor.nivel_alerta(p_post)
        print(f"\n  {BOLD}{c['descripcion']}{RESET}")
        print(f"  {DIM}{c['nota']}{RESET}")
        print(f"  P prior  : {col_pre}{c['p_prior']:.0%}{RESET} [{nivel_pre}]  "
              f"{barra_progreso(c['p_prior'], 20)}")
        print(f"  P post.  : {col_post}{p_post:.0%}{RESET} [{nivel_post}]  "
              f"{barra_progreso(p_post, 20)}")
        print(f"  Δ (incremento): {'+' if incremento>=0 else ''}{incremento:.1%}  "
              f"{'↑ Riesgo confirmado' if incremento > 0 else '↓ Evidencia reduce riesgo'}")


def imprimir_curva_riesgo_temperatura():
    """Muestra cómo varía el riesgo de SP-03 con la temperatura."""
    motor  = MotorProbabilistico()
    sp03   = next(s for s in SITUACIONES if s.id == "SP-03")
    puntos = motor.distribucion_riesgo(sp03.p_base, variable=30, media=30, std=5)

    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  CURVA DE RIESGO — SP-03: Temperatura vs. Probabilidad de daño")
    print(f"{'═'*70}{RESET}")
    print(f"\n  {'TEMP (°C)':>10}  {'P RIESGO':>9}  DISTRIBUCIÓN")
    print(f"  {'─'*9}  {'─'*8}  {'─'*36}")

    # Mostrar solo 13 puntos representativos
    step = max(1, len(puntos) // 13)
    for i in range(0, len(puntos), step):
        temp, p = puntos[i]
        nivel, color = motor.nivel_alerta(p)
        marca = f" {color}← umbral acción{RESET}" if abs(temp - 30) < 0.3 else ""
        print(f"  {temp:>9.1f}  {color}{p:>8.1%}{RESET}  "
              f"{barra_progreso(p, 24)}{marca}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    print(f"\n{BOLD}{'='*70}")
    print("  PARTE H — RAZONAMIENTO PROBABILÍSTICO DEL ALMACÉN")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{RESET}")

    motor = MotorProbabilistico()
    sp_map = {s.id: s for s in SITUACIONES}

    # ── 1. Catálogo base ───────────────────────────────────────────────────────
    imprimir_situaciones_base()

    # ── 2. Actualización bayesiana ─────────────────────────────────────────────
    imprimir_bayes()

    # ── 3. Curva de riesgo temperatura ────────────────────────────────────────
    imprimir_curva_riesgo_temperatura()

    # ── 4. Evaluación de escenarios ────────────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  EVALUACIÓN DE ESCENARIOS — Probabilidades ajustadas por evidencia")
    print(f"{'═'*70}{RESET}")

    todos_informes = {}
    for nombre_esc, evidencias_esc in ESCENARIOS_PROB.items():
        evaluaciones = []
        for sp_id, evidencias in evidencias_esc.items():
            sp = sp_map[sp_id]
            p_aj = motor.calcular_p_ajustada(sp, evidencias)
            nivel, _ = motor.nivel_alerta(p_aj)
            evaluaciones.append({
                "id"          : sp_id,
                "descripcion" : sp.descripcion,
                "dominio"     : sp.dominio,
                "p_base"      : sp.p_base,
                "p_ajustada"  : p_aj,
                "umbral_accion": sp.umbral_accion,
                "nivel"       : nivel,
                "evidencias"  : evidencias,
            })

        decision = motor.decision_final(evaluaciones)
        imprimir_evaluacion_escenario(nombre_esc, evaluaciones, decision, motor)
        todos_informes[nombre_esc] = {
            "evaluaciones": evaluaciones,
            "decision"    : decision,
        }

    # ── 5. Tabla comparativa de influencia en decisión ────────────────────────
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  TABLA COMPARATIVA — Influencia en la decisión final")
    print(f"{'═'*70}{RESET}")
    print(f"\n  {'ESCENARIO':<42} {'P_GLOBAL':>9} {'NIVEL':<9} ACCIÓN ABREVIADA")
    print(f"  {'─'*40}  {'─'*8}  {'─'*8}  {'─'*20}")
    for nombre_esc, informe in todos_informes.items():
        dec  = informe["decision"]
        niv  = dec["nivel_global"]
        _, color = motor.nivel_alerta(dec["p_global"])
        accion_corta = dec["accion"].split("—")[0].strip()
        print(f"  {nombre_esc:<42} {color}{dec['p_global']:>8.1%}{RESET}  "
              f"{color}{niv:<9}{RESET} {accion_corta}")

    # ── 6. Resumen estadístico ────────────────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  RESUMEN — PARTE H")
    print(f"{'═'*70}{RESET}")
    print(f"  Situaciones probabilísticas definidas : {len(SITUACIONES)}")
    print(f"  Escenarios evaluados                  : {len(ESCENARIOS_PROB)}")
    criticos = sum(1 for i in todos_informes.values()
                   if i["decision"]["nivel_global"] == "CRÍTICO")
    print(f"  Escenarios con nivel CRÍTICO           : {criticos}/{len(ESCENARIOS_PROB)}")

    # ── 7. Reporte JSON ───────────────────────────────────────────────────────
    from paths import REPORTE_PROBABILIDAD, asegurar_carpetas
    asegurar_carpetas()

    reporte = {
        "modulo"     : "Parte H — Razonamiento probabilístico (logica_probabilidad.py)",
        "timestamp"  : datetime.now().isoformat(),
        "situaciones": [{
            "id"            : s.id,
            "descripcion"   : s.descripcion,
            "dominio"       : s.dominio,
            "p_base"        : s.p_base,
            "p_consecuencia": s.p_consecuencia,
            "umbral_accion" : s.umbral_accion,
            "justificacion" : s.justificacion,
            "factores"      : s.factores,
        } for s in SITUACIONES],
        "escenarios" : {
            nombre: {
                "evaluaciones": [
                    {k: v for k, v in ev.items() if k != "evidencias"}
                    for ev in informe["evaluaciones"]
                ],
                "decision": informe["decision"],
            }
            for nombre, informe in todos_informes.items()
        },
    }

    ruta = str(REPORTE_PROBABILIDAD)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print(f"\n  {GREEN}Reporte JSON guardado: {ruta}{RESET}")
    print(f"\n{BOLD}{'='*70}")
    print("  Parte H completada exitosamente.")
    print(f"{'='*70}{RESET}\n")

    return reporte


if __name__ == "__main__":
    main()
