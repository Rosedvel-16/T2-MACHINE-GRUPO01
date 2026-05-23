"""
===============================================================================
PARTE G — LÓGICA GENERAL Y LÓGICA PROPOSICIONAL
Sistema Inteligente Multiagente para Almacén Logístico
===============================================================================

Este módulo implementa la representación formal del problema de supervisión
del almacén mediante lógica proposicional clásica.

Contenido:
  1. Definición de proposiciones atómicas (≥ 5)
  2. Expresiones lógicas compuestas (≥ 4)
  3. Motor de inferencia por tabla de verdad
  4. Evaluador de escenarios reales del almacén
  5. Trazado de cadenas de razonamiento (proof trace)
===============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import product
import json
from datetime import datetime

# ── Colores consola ────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROPOSICIONES ATÓMICAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Proposicion:
    """Representa una proposición atómica del dominio del almacén."""
    simbolo    : str   # Letra o código corto (ej. "P", "Q", "T_ALTA")
    descripcion: str   # Qué representa en lenguaje natural
    dominio    : str   # Subsistema al que pertenece

# Catálogo de proposiciones atómicas (≥ 5)
PROPOSICIONES: Dict[str, Proposicion] = {
    "P": Proposicion("P", "La vibración en la estantería es anómala",         "sensores"),
    "Q": Proposicion("Q", "Existe riesgo mecánico en la zona",                 "diagnóstico"),
    "R": Proposicion("R", "Se detiene la operación de carga/descarga",         "control"),
    "T": Proposicion("T", "La temperatura del almacén es alta (> 30 °C)",      "sensores"),
    "H": Proposicion("H", "La humedad ambiental es excesiva (> 80%)",          "sensores"),
    "A": Proposicion("A", "Se emite alerta ambiental al supervisor",           "alertas"),
    "D": Proposicion("D", "El paquete inspeccionado presenta daño visible",    "visión"),
    "M": Proposicion("M", "El paquete requiere revisión manual",               "control"),
    "Z": Proposicion("Z", "La zona de carga está saturada (> 90% ocupación)",  "sensores"),
    "O": Proposicion("O", "La ruta de acceso está obstruida",                  "visión"),
    "C": Proposicion("C", "Se ordena la reubicación de carga",                 "control"),
    "E": Proposicion("E", "El pedido tiene prioridad alta (cliente VIP)",      "logística"),
    "L": Proposicion("L", "La zona de despacho está libre (< 40% ocupación)", "sensores"),
    "S": Proposicion("S", "Se recomienda despacho inmediato",                  "logística"),
    "N": Proposicion("N", "Nivel de riesgo operativo global es CRÍTICO",       "diagnóstico"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPRESIONES LÓGICAS COMPUESTAS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpresionLogica:
    """Fórmula proposicional con metadatos del dominio."""
    id         : str
    formula_str: str           # Representación textual legible
    descripcion: str           # Significado operativo
    tipo       : str           # "regla", "alerta", "condición"
    evaluador  : callable      # Función lambda que recibe dict{str:bool} → bool
    antecedentes: List[str]    # Variables de entrada
    consecuente : str          # Variable de salida / conclusión


EXPRESIONES: List[ExpresionLogica] = [

    # ── EL-01: Riesgo mecánico ─────────────────────────────────────────────────
    ExpresionLogica(
        id          = "EL-01",
        formula_str = "P → Q",
        descripcion = "Si la vibración es anómala, entonces existe riesgo mecánico.",
        tipo        = "regla",
        evaluador   = lambda v: (not v["P"]) or v["Q"],
        antecedentes= ["P"],
        consecuente = "Q",
    ),

    # ── EL-02: Parada de emergencia ────────────────────────────────────────────
    ExpresionLogica(
        id          = "EL-02",
        formula_str = "Q ∧ R → N",
        descripcion = "Riesgo mecánico activo Y operación detenida → nivel crítico global.",
        tipo        = "alerta",
        evaluador   = lambda v: (not (v["Q"] and v["R"])) or v["N"],
        antecedentes= ["Q", "R"],
        consecuente = "N",
    ),

    # ── EL-03: Alerta ambiental ────────────────────────────────────────────────
    ExpresionLogica(
        id          = "EL-03",
        formula_str = "T ∧ H → A",
        descripcion = "Temperatura alta Y humedad excesiva → emitir alerta ambiental.",
        tipo        = "regla",
        evaluador   = lambda v: (not (v["T"] and v["H"])) or v["A"],
        antecedentes= ["T", "H"],
        consecuente = "A",
    ),

    # ── EL-04: Revisión manual de paquete ──────────────────────────────────────
    ExpresionLogica(
        id          = "EL-04",
        formula_str = "D → M",
        descripcion = "Paquete dañado detectado → requiere revisión manual.",
        tipo        = "regla",
        evaluador   = lambda v: (not v["D"]) or v["M"],
        antecedentes= ["D"],
        consecuente = "M",
    ),

    # ── EL-05: Reubicación de carga ────────────────────────────────────────────
    ExpresionLogica(
        id          = "EL-05",
        formula_str = "Z ∧ O → C",
        descripcion = "Zona saturada Y ruta obstruida → ordenar reubicación de carga.",
        tipo        = "regla",
        evaluador   = lambda v: (not (v["Z"] and v["O"])) or v["C"],
        antecedentes= ["Z", "O"],
        consecuente = "C",
    ),

    # ── EL-06: Despacho inmediato ──────────────────────────────────────────────
    ExpresionLogica(
        id          = "EL-06",
        formula_str = "E ∧ L → S",
        descripcion = "Pedido prioritario Y zona libre → recomendar despacho inmediato.",
        tipo        = "regla",
        evaluador   = lambda v: (not (v["E"] and v["L"])) or v["S"],
        antecedentes= ["E", "L"],
        consecuente = "S",
    ),

    # ── EL-07: Riesgo ambiental escalado ──────────────────────────────────────
    ExpresionLogica(
        id          = "EL-07",
        formula_str = "A ∨ N → R",
        descripcion = "Alerta ambiental O nivel crítico → detener operaciones.",
        tipo        = "alerta",
        evaluador   = lambda v: (not (v["A"] or v["N"])) or v["R"],
        antecedentes= ["A", "N"],
        consecuente = "R",
    ),

    # ── EL-08: Doble condición de seguridad ────────────────────────────────────
    ExpresionLogica(
        id          = "EL-08",
        formula_str = "¬M ∧ ¬O → S",
        descripcion = "Sin revisión pendiente Y sin obstrucción → zona apta para despacho.",
        tipo        = "condición",
        evaluador   = lambda v: (not ((not v["M"]) and (not v["O"]))) or v["S"],
        antecedentes= ["M", "O"],
        consecuente = "S",
    ),

    # ── EL-09: Bicondicional riesgo-parada ─────────────────────────────────────
    ExpresionLogica(
        id          = "EL-09",
        formula_str = "Q ↔ R   (bajo P activo)",
        descripcion = "Riesgo mecánico si y sólo si la operación está detenida (cuando vibración anómala).",
        tipo        = "condición",
        evaluador   = lambda v: (v["Q"] == v["R"]) if v["P"] else True,
        antecedentes= ["P", "Q", "R"],
        consecuente = "Q↔R",
    ),

    # ── EL-10: Cascada completa de riesgo ─────────────────────────────────────
    ExpresionLogica(
        id          = "EL-10",
        formula_str = "(P ∨ T) ∧ (H ∨ D) → N",
        descripcion = "Combinación de señales críticas de sensor y visión → riesgo global.",
        tipo        = "alerta",
        evaluador   = lambda v: (not ((v["P"] or v["T"]) and (v["H"] or v["D"]))) or v["N"],
        antecedentes= ["P", "T", "H", "D"],
        consecuente = "N",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MOTOR DE EVALUACIÓN LÓGICA
# ═══════════════════════════════════════════════════════════════════════════════

class MotorLogico:
    """
    Motor de inferencia proposicional.

    Funcionalidades:
      - Evaluar todas las expresiones dado un estado del mundo (dict de valores).
      - Generar tabla de verdad para expresiones seleccionadas.
      - Identificar violaciones (fórmulas evaluadas como FALSAS).
      - Inferir consecuentes mediante Modus Ponens.
    """

    def __init__(self):
        self.proposiciones = PROPOSICIONES
        self.expresiones   = EXPRESIONES

    # ── Evaluación de un escenario ─────────────────────────────────────────────
    def evaluar_escenario(self, valores: Dict[str, bool]) -> Dict:
        """
        Evalúa todas las expresiones lógicas con los valores del mundo dado.
        Retorna dict con resultados, violaciones e inferencias.
        """
        resultados    = {}
        violaciones   = []
        inferencias   = {}

        for expr in self.expresiones:
            try:
                resultado = expr.evaluador(valores)
            except KeyError as e:
                resultado = None  # variable no definida en el escenario
            resultados[expr.id] = resultado
            if resultado is False:
                violaciones.append(expr)

        # Inferencia hacia adelante (Modus Ponens simple)
        for expr in self.expresiones:
            if expr.tipo == "regla":
                antec_true = all(valores.get(a, False) for a in expr.antecedentes)
                if antec_true and "↔" not in expr.formula_str:
                    inferencias[expr.consecuente] = True

        return {
            "resultados"  : resultados,
            "violaciones" : violaciones,
            "inferencias" : inferencias,
            "consistente" : len(violaciones) == 0,
        }

    # ── Tabla de verdad ────────────────────────────────────────────────────────
    def tabla_de_verdad(self, id_expresion: str, variables: List[str]) -> List[dict]:
        """Genera tabla de verdad para una expresión dada."""
        expr = next((e for e in self.expresiones if e.id == id_expresion), None)
        if expr is None:
            raise ValueError(f"Expresión {id_expresion} no encontrada")

        filas = []
        for combo in product([False, True], repeat=len(variables)):
            val = dict(zip(variables, combo))
            # Rellenar el resto con False por defecto
            val_full = {k: False for k in self.proposiciones}
            val_full.update(val)
            resultado = expr.evaluador(val_full)
            filas.append({**val, "resultado": resultado})
        return filas

    # ── Modus Ponens explícito ─────────────────────────────────────────────────
    def modus_ponens(self, antecedente: bool, implicacion_id: str,
                     valores: Dict[str, bool]) -> Tuple[bool, str]:
        """
        Aplica Modus Ponens: si P es verdadero y P→Q es válida, concluye Q.
        Retorna (conclusión, traza textual).
        """
        expr = next((e for e in self.expresiones if e.id == implicacion_id), None)
        if expr is None:
            return False, f"Expresión {implicacion_id} no encontrada."

        if antecedente:
            val_full = {k: False for k in self.proposiciones}
            val_full.update(valores)
            val_full[expr.antecedentes[0]] = True
            resultado = expr.evaluador(val_full)
            traza = (f"  Modus Ponens [{expr.id}]: "
                     f"{expr.antecedentes[0]}=True, fórmula '{expr.formula_str}' "
                     f"→ {expr.consecuente}={resultado}")
            return resultado, traza
        return False, f"  Antecedente falso — no se puede aplicar MP en {implicacion_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ESCENARIOS DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════

ESCENARIOS = {
    "ESC-1: Falla mecánica activa": {
        "P": True,  "Q": True,  "R": True,  "T": False, "H": False,
        "A": False, "D": False, "M": False, "Z": False, "O": False,
        "C": False, "E": False, "L": False, "S": False, "N": True,
    },
    "ESC-2: Alerta ambiental (calor+humedad)": {
        "P": False, "Q": False, "R": False, "T": True,  "H": True,
        "A": True,  "D": False, "M": False, "Z": False, "O": False,
        "C": False, "E": False, "L": True,  "S": False, "N": False,
    },
    "ESC-3: Paquete dañado + zona saturada": {
        "P": False, "Q": False, "R": False, "T": False, "H": False,
        "A": False, "D": True,  "M": True,  "Z": True,  "O": True,
        "C": True,  "E": True,  "L": False, "S": False, "N": False,
    },
    "ESC-4: Despacho expedito exitoso": {
        "P": False, "Q": False, "R": False, "T": False, "H": False,
        "A": False, "D": False, "M": False, "Z": False, "O": False,
        "C": False, "E": True,  "L": True,  "S": True,  "N": False,
    },
    "ESC-5: Crisis múltiple": {
        "P": True,  "Q": True,  "R": True,  "T": True,  "H": True,
        "A": True,  "D": True,  "M": True,  "Z": True,  "O": True,
        "C": True,  "E": False, "L": False, "S": False, "N": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FUNCIONES DE PRESENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def imprimir_catalogo():
    """Muestra todas las proposiciones y expresiones definidas."""
    print(f"\n{CYAN}{BOLD}{'─'*70}")
    print("  CATÁLOGO DE PROPOSICIONES ATÓMICAS")
    print(f"{'─'*70}{RESET}")
    print(f"  {'SÍM.':<8} {'DOMINIO':<14} DESCRIPCIÓN")
    print(f"  {'─'*6}  {'─'*12}  {'─'*44}")
    for sim, prop in PROPOSICIONES.items():
        print(f"  {BOLD}{sim:<8}{RESET} {DIM}{prop.dominio:<14}{RESET} {prop.descripcion}")

    print(f"\n{CYAN}{BOLD}{'─'*70}")
    print("  EXPRESIONES LÓGICAS COMPUESTAS")
    print(f"{'─'*70}{RESET}")
    for expr in EXPRESIONES:
        tipo_color = RED if expr.tipo == "alerta" else (YELLOW if expr.tipo == "regla" else GREEN)
        print(f"\n  {BOLD}[{expr.id}]{RESET}  {tipo_color}{expr.formula_str}{RESET}")
        print(f"           {expr.descripcion}")
        print(f"           {DIM}Tipo: {expr.tipo} | "
              f"Ant: {expr.antecedentes} → Con: {expr.consecuente}{RESET}")


def imprimir_tabla_de_verdad(motor: MotorLogico, id_expr: str, vars_: List[str]):
    """Imprime la tabla de verdad de una expresión específica."""
    expr = next((e for e in EXPRESIONES if e.id == id_expr), None)
    filas = motor.tabla_de_verdad(id_expr, vars_)

    print(f"\n  {CYAN}{BOLD}Tabla de verdad — {id_expr}: {expr.formula_str}{RESET}")
    # Encabezado
    header = "  " + " | ".join(f"{v:^5}" for v in vars_) + f" | {'RESULT':^6}"
    print(f"  {DIM}{'─'*len(header)}{RESET}")
    print(header)
    print(f"  {DIM}{'─'*len(header)}{RESET}")
    for fila in filas:
        vals_str = " | ".join(f"{'V':^5}" if fila[v] else f"{'F':^5}" for v in vars_)
        res = fila["resultado"]
        res_str = f"{GREEN}  V  {RESET}" if res else f"{RED}  F  {RESET}"
        print(f"  {vals_str} | {res_str}")
    print(f"  {DIM}{'─'*len(header)}{RESET}")


def imprimir_evaluacion_escenario(nombre: str, valores: Dict[str, bool],
                                   informe: Dict, motor: MotorLogico):
    """Imprime el análisis completo de un escenario."""
    print(f"\n{YELLOW}{BOLD}  ▶ ESCENARIO: {nombre}{RESET}")

    # Estado del mundo
    activos = [f"{BOLD}{k}{RESET}" for k, v in valores.items() if v]
    inact   = [f"{DIM}{k}{RESET}"  for k, v in valores.items() if not v]
    print(f"  Proposiciones TRUE  : {', '.join(activos) if activos else '(ninguna)'}")
    print(f"  Proposiciones FALSE : {', '.join(inact) if inact else '(ninguna)'}")

    # Resultados de expresiones
    print(f"\n  {'EXPR':<8} {'FÓRMULA':<28} {'RESULTADO'}")
    print(f"  {'─'*6}  {'─'*26}  {'─'*10}")
    for expr in EXPRESIONES:
        res = informe["resultados"].get(expr.id)
        if res is True:
            res_str = f"{GREEN}VÁLIDA{RESET}"
        elif res is False:
            res_str = f"{RED}VIOLADA{RESET}"
        else:
            res_str = f"{DIM}N/A{RESET}"
        print(f"  {expr.id:<8} {expr.formula_str:<28} {res_str}")

    # Inferencias
    if informe["inferencias"]:
        print(f"\n  {GREEN}Inferencias derivadas (Modus Ponens):{RESET}")
        for cons, val in informe["inferencias"].items():
            desc = PROPOSICIONES.get(cons, None)
            desc_str = desc.descripcion if desc else cons
            print(f"    → {BOLD}{cons}{RESET} = {val}  |  {desc_str}")

    # Violaciones
    if informe["violaciones"]:
        print(f"\n  {RED}Expresiones VIOLADAS (inconsistencias):{RESET}")
        for expr in informe["violaciones"]:
            print(f"    ✗ [{expr.id}] {expr.formula_str}  — {expr.descripcion}")
    else:
        print(f"\n  {GREEN}✓ Escenario consistente — todas las reglas satisfechas{RESET}")

    print(f"  {'─'*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    print(f"\n{BOLD}{'='*70}")
    print("  PARTE G — LÓGICA PROPOSICIONAL DEL ALMACÉN")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{RESET}")

    motor = MotorLogico()

    # ── 1. Catálogo completo ───────────────────────────────────────────────────
    imprimir_catalogo()

    # ── 2. Tablas de verdad seleccionadas ─────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  TABLAS DE VERDAD (expresiones clave)")
    print(f"{'═'*70}{RESET}")

    imprimir_tabla_de_verdad(motor, "EL-01", ["P", "Q"])
    imprimir_tabla_de_verdad(motor, "EL-03", ["T", "H", "A"])
    imprimir_tabla_de_verdad(motor, "EL-05", ["Z", "O", "C"])
    imprimir_tabla_de_verdad(motor, "EL-06", ["E", "L", "S"])

    # ── 3. Modus Ponens explícito ──────────────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  DEMOSTRACIÓN DE MODUS PONENS")
    print(f"{'═'*70}{RESET}")

    print(f"\n  {YELLOW}Caso 1: P=True (vibración anómala) → ¿Q válida?{RESET}")
    resultado_mp, traza = motor.modus_ponens(True, "EL-01", {"P": True})
    print(traza)

    print(f"\n  {YELLOW}Caso 2: T=True, H=True (calor + humedad) → ¿A válida?{RESET}")
    _, traza2 = motor.modus_ponens(True, "EL-03", {"T": True, "H": True})
    print(traza2.replace("antecedentes[0]", "T+H"))

    print(f"\n  {YELLOW}Caso 3: E=True, L=True (pedido VIP + zona libre) → ¿S válida?{RESET}")
    _, traza3 = motor.modus_ponens(True, "EL-06", {"E": True, "L": True})
    print(traza3)

    # ── 4. Evaluación de escenarios reales ────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  EVALUACIÓN DE ESCENARIOS DEL ALMACÉN")
    print(f"{'═'*70}{RESET}")

    informes = {}
    for nombre, valores in ESCENARIOS.items():
        informe = motor.evaluar_escenario(valores)
        informes[nombre] = informe
        imprimir_evaluacion_escenario(nombre, valores, informe, motor)

    # ── 5. Resumen estadístico ─────────────────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═'*70}")
    print("  RESUMEN — PARTE G")
    print(f"{'═'*70}{RESET}")
    print(f"  Proposiciones atómicas definidas : {len(PROPOSICIONES)}")
    print(f"  Expresiones lógicas compuestas   : {len(EXPRESIONES)}")
    print(f"  Escenarios evaluados             : {len(ESCENARIOS)}")
    consistentes = sum(1 for i in informes.values() if i["consistente"])
    print(f"  Escenarios consistentes          : {consistentes}/{len(ESCENARIOS)}")

    # ── 6. Exportar reporte ────────────────────────────────────────────────────
    reporte = {
        "modulo"      : "Parte G — Lógica proposicional (logica_proposicional.py)",
        "timestamp"   : datetime.now().isoformat(),
        "proposiciones": {k: {"simbolo": v.simbolo, "descripcion": v.descripcion,
                               "dominio": v.dominio}
                          for k, v in PROPOSICIONES.items()},
        "expresiones" : [{
            "id"         : e.id,
            "formula"    : e.formula_str,
            "descripcion": e.descripcion,
            "tipo"       : e.tipo,
        } for e in EXPRESIONES],
        "escenarios"  : {
            nombre: {
                "valores"      : {k: bool(v) for k, v in vals.items()},
                "consistente"  : informes[nombre]["consistente"],
                "inferencias"  : {k: bool(v) for k, v in informes[nombre]["inferencias"].items()},
                "n_violaciones": len(informes[nombre]["violaciones"]),
            }
            for nombre, vals in ESCENARIOS.items()
        },
    }

    import os
    os.makedirs("images_output", exist_ok=True)
    ruta = "images_output/reporte_logica.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print(f"\n  {GREEN}Reporte JSON guardado: {ruta}{RESET}")
    print(f"\n{BOLD}{'='*70}")
    print("  Parte G completada exitosamente.")
    print(f"{'='*70}{RESET}\n")

    return reporte


if __name__ == "__main__":
    main()
