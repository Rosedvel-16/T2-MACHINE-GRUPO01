import os
import cv2
import numpy as np
import streamlit as st

import vision_cv
import logica_proposicional as lp
import logica_probabilidad as lprob
import sistema_experto as se
import agentes as ag
import concurrencia as conc
import optimizacion_despacho as opt
from paths import CSV_SENSORES, asegurar_carpetas

asegurar_carpetas()


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    # OpenCV carga en BGR; Streamlit espera RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image(path: str) -> np.ndarray:
    # Carga robusta de imagenes, con fallback para rutas Unicode en Windows.
    img = cv2.imread(path)
    if img is None:
        # Fallback para rutas con caracteres Unicode en Windows
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img


def show_image(title: str, img_path: str):
    # Muestra una imagen con titulo en la interfaz.
    st.subheader(title)
    img = read_image(img_path)
    st.image(bgr_to_rgb(img), use_container_width=True)


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

        :root {
            --bg-0: #0a0f14;
            --bg-1: #111923;
            --bg-2: #0e141d;
            --steel: #9fb2c6;
            --ink: #e7edf5;
            --cyan: #23f0d1;
            --amber: #ffb347;
            --red: #ff5a5f;
            --grid: rgba(255, 255, 255, 0.04);
        }

        .stApp {
            background:
                radial-gradient(1200px 700px at 85% -10%, rgba(35, 240, 209, 0.10), transparent 60%),
                radial-gradient(900px 600px at 10% 110%, rgba(255, 179, 71, 0.12), transparent 55%),
                linear-gradient(130deg, var(--bg-0), var(--bg-1) 55%, var(--bg-2));
            color: var(--ink);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.04em;
        }

        .hero {
            padding: 1.5rem 2rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            background: linear-gradient(130deg, rgba(20, 28, 39, 0.85), rgba(12, 18, 27, 0.85));
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
        }

        .subtle {
            color: var(--steel);
            font-size: 0.95rem;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
        }

        .kpi-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(9, 14, 21, 0.7);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
        }

        .kpi-label {
            color: var(--steel);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .kpi-value {
            font-size: 1.8rem;
            margin-top: 0.4rem;
            font-weight: 600;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: rgba(35, 240, 209, 0.15);
            color: var(--cyan);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .alert-card {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(12, 18, 27, 0.7);
            margin-bottom: 0.8rem;
        }

        .divider-line {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
            margin: 1.2rem 0;
        }

        .stButton button {
            border-radius: 999px;
            border: 1px solid rgba(35, 240, 209, 0.45);
            background: rgba(35, 240, 209, 0.08);
            color: var(--cyan);
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .stButton button:hover {
            border-color: var(--cyan);
            box-shadow: 0 0 18px rgba(35, 240, 209, 0.25);
        }

        .dataframe, .stTable {
            background: rgba(10, 15, 20, 0.7);
        }

        @media (max-width: 1200px) {
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 768px) {
            .kpi-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str, tone: str = "cyan"):
    color = "var(--cyan)" if tone == "cyan" else "var(--amber)" if tone == "amber" else "var(--red)"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{value}</div>
            <div class="subtle">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_kpi_state() -> dict:
    riesgo = float(np.clip(np.random.normal(0.62, 0.14), 0.05, 0.98))
    ocupacion = int(np.clip(np.random.normal(78, 12), 40, 99))
    alertas = int(np.clip(np.interp(riesgo, [0, 1], [1, 8]) + np.random.randint(0, 3), 0, 12))
    concurrencia = int(np.clip(np.random.normal(18, 5), 6, 32))
    return {
        "riesgo": riesgo,
        "ocupacion": ocupacion,
        "alertas": alertas,
        "concurrencia": concurrencia,
        "timestamp": "actualizado",
    }


def get_state():
    if "kpi" not in st.session_state:
        st.session_state["kpi"] = generate_kpi_state()
    if "signals" not in st.session_state:
        st.session_state["signals"] = build_signal_window()
    if "concurrency" not in st.session_state:
        st.session_state["concurrency"] = build_concurrency_window()


def build_signal_window():
    steps = 60
    return {
        "vibracion": np.cumsum(np.random.randn(steps)) + 10,
        "temperatura": np.cumsum(np.random.randn(steps)) + 28,
        "humedad": np.cumsum(np.random.randn(steps)) + 70,
    }


def build_concurrency_window():
    return {
        "cola_picking": int(np.random.randint(12, 45)),
        "cola_packing": int(np.random.randint(8, 32)),
        "robots_activos": int(np.random.randint(4, 14)),
        "turnos_activos": int(np.random.randint(2, 6)),
    }


def render_header():
    st.markdown(
        """
        <div class="hero">
            <div class="chip">Industrial Futurista</div>
            <h1>Sistema Inteligente Multiagente</h1>
            <p class="subtle">Demo academica para supervisores logisticos · Streamlit UI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard():
    st.subheader("Dashboard principal")
    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.markdown("Estado general del almacen y telemetria operacional.")
    with col_b:
        if st.button("Refrescar estado", key="btn_refresh"):
            st.session_state["kpi"] = generate_kpi_state()

    kpi = st.session_state["kpi"]
    st.markdown(
        """
        <div class="kpi-grid">
        """,
        unsafe_allow_html=True,
    )
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        kpi_card("Riesgo global", f"{kpi['riesgo']:.2f}", "Noisy-OR fusion", "cyan")
    with kpi_cols[1]:
        kpi_card("Ocupacion", f"{kpi['ocupacion']}%", "Sensores pasillo", "amber")
    with kpi_cols[2]:
        kpi_card("Alertas activas", f"{kpi['alertas']}", "Ultimos 15 min", "red")
    with kpi_cols[3]:
        kpi_card("Concurrencia", f"{kpi['concurrencia']} hilos", "Estado scheduler", "cyan")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("**Estado del sistema**")
        st.progress(min(int(kpi["riesgo"] * 100), 100))
        st.write("Motor experto, agentes y vision sincronizados. Ultimo ciclo exitoso.")
    with col2:
        st.markdown("**Acciones rapidas**")
        st.button("Disparar ciclo multiagente", key="btn_cycle")
        st.button("Recalcular decisiones", key="btn_decision")
        st.button("Exportar reporte", key="btn_export")


def render_agents():
    st.subheader("Panel de agentes")
    st.write("Vista consolidada de los agentes y su estado operativo.")

    info_agentes = [
        {
            "Agente": "Sensor",
            "Estado": "Activo",
            "Rol": "Lecturas ambientales y vibracion",
            "Latencia(ms)": 120,
        },
        {
            "Agente": "AnalizadorSenales",
            "Estado": "Activo",
            "Rol": "Umbralizacion + conversion a hechos",
            "Latencia(ms)": 210,
        },
        {
            "Agente": "AnalizadorImagenes",
            "Estado": "En espera",
            "Rol": "Vision CV + reporte JSON",
            "Latencia(ms)": 340,
        },
        {
            "Agente": "Decisor",
            "Estado": "Activo",
            "Rol": "Forward chaining + prioridades",
            "Latencia(ms)": 160,
        },
        {
            "Agente": "Coordinador",
            "Estado": "Activo",
            "Rol": "Fusiones y plan global",
            "Latencia(ms)": 190,
        },
    ]
    st.table(info_agentes)

    with st.expander("Ejecutar pipeline multiagente"):
        col_a, col_b = st.columns(2)
        with col_a:
            ciclos_n = st.slider("Numero de ciclos", 1, 10, 3)
        with col_b:
            usar_vision = st.checkbox("Incluir vision", value=True)

        if st.button("Ejecutar pipeline", key="btn_agt"):
            csv_path = str(CSV_SENSORES) if CSV_SENSORES.exists() else None

            sistema = ag.SistemaMultiAgente(csv_sensores=csv_path)
            resultados_ciclos = []
            for _ in range(ciclos_n):
                resultados_ciclos.append(sistema.ciclo(incluir_vision=usar_vision))

            st.success(
                f"Pipeline ejecutado: {ciclos_n} ciclos, {len(sistema.historial)} mensajes."
            )
            ultimo = resultados_ciclos[-1]["plan_global"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Nivel de riesgo", ultimo.get("nivel_global", "—"))
            col2.metric("Riesgo global", f"{ultimo.get('riesgo_global_pct', 0)}%")
            col3.metric("Mensajes", len(sistema.historial))

            st.markdown(f"**Directiva:** {ultimo.get('directiva_principal', '—')}")


def render_alerts():
    st.subheader("Centro de alertas")
    st.write("Canal priorizado de incidentes y recomendaciones operativas.")

    kpi = st.session_state["kpi"]
    nivel = "ALTO" if kpi["riesgo"] > 0.75 else "MEDIO" if kpi["riesgo"] > 0.55 else "BAJO"
    st.markdown(
        f"""
        <div class="alert-card">
            <strong>Nivel global:</strong> {nivel} · Riesgo {kpi['riesgo']:.2f}
            <div class="subtle">Fusion de sensores + vision + reglas expertas.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alertas = [
        "Riesgo mecanico elevado en pasillo 3. Revisar vibracion > 3 mm/s.",
        "Saturacion de zona de despacho. Posible bloqueo en ruta B.",
        "Humedad alta sostenida. Verificar climatizacion del almacén.",
    ]
    for alerta in alertas:
        st.markdown(
            f"""
            <div class="alert-card">
                <div class="subtle">PRIORIDAD</div>
                <strong>{alerta}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_realtime():
    st.subheader("Graficos en tiempo real")
    st.write("Ventana dinamica de sensores: vibracion, temperatura y humedad.")
    if st.button("Simular nueva ventana", key="btn_signals"):
        st.session_state["signals"] = build_signal_window()

    signals = st.session_state["signals"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.line_chart(signals["vibracion"], height=220)
        st.caption("Vibracion (mm/s)")
    with col2:
        st.line_chart(signals["temperatura"], height=220)
        st.caption("Temperatura (C)")
    with col3:
        st.line_chart(signals["humedad"], height=220)
        st.caption("Humedad (%)")


def render_concurrency():
    st.subheader("Panel de concurrencia")
    st.write("Monitoreo de colas, procesos activos y turnos en ejecucion.")
    if st.button("Actualizar concurrencia", key="btn_conc"):
        st.session_state["concurrency"] = build_concurrency_window()

    conc = st.session_state["concurrency"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cola picking", conc["cola_picking"])
    col2.metric("Cola packing", conc["cola_packing"])
    col3.metric("Robots activos", conc["robots_activos"])
    col4.metric("Turnos activos", conc["turnos_activos"])

    st.progress(min(conc["cola_picking"], 100) / 100)
    st.caption("Carga relativa de la cola principal")


def render_gallery():
    st.subheader("Galeria de imagenes procesadas")
    st.write("Resultados de vision CV y procesamiento de imagenes.")

    jpgs = vision_cv.listar_jpgs()
    if len(jpgs) >= 3:
        st.markdown("**Imagenes base detectadas:**")
        st.write(", ".join(jpgs))
        if st.button("Reprocesar imagenes", key="btn_reprocess"):
            with st.spinner("Procesando..."):
                vision_cv.procesar_paquete_danado(jpgs[0])
                vision_cv.procesar_paquete_buen_estado(jpgs[1])
                vision_cv.procesar_zona_obstruida(jpgs[2])
            st.success("Procesamiento completado.")
    else:
        st.warning("No se encontraron 3 imagenes base para reprocesar.")

    archivos = [
        f for f in os.listdir(vision_cv.RUTA_SALIDA)
        if f.lower().endswith(".jpg")
    ]
    archivos = sorted(archivos)
    if not archivos:
        st.info("Sin imagenes procesadas para mostrar.")
        return

    cols = st.columns(3)
    for idx, img_name in enumerate(archivos):
        with cols[idx % 3]:
            img_path = os.path.join(vision_cv.RUTA_SALIDA, img_name)
            st.image(read_image(img_path), use_container_width=True, caption=img_name)


def render_reasoning():
    st.subheader("Laboratorio de razonamiento")
    st.write("Explora logica proposicional y probabilistica para la demo academica.")

    with st.expander("Logica proposicional"):
        st.subheader("Proposiciones")
        prop_rows = [
            {"Simbolo": p.simbolo, "Descripcion": p.descripcion, "Dominio": p.dominio}
            for p in lp.PROPOSICIONES.values()
        ]
        st.table(prop_rows)

        st.subheader("Expresiones logicas")
        expr_rows = [
            {"ID": e.id, "Formula": e.formula_str, "Tipo": e.tipo, "Descripcion": e.descripcion}
            for e in lp.EXPRESIONES
        ]
        st.table(expr_rows)

        st.subheader("Evaluacion de escenarios")
        motor = lp.MotorLogico()
        escenario = st.selectbox("Escenario logico", list(lp.ESCENARIOS.keys()))
        valores = lp.ESCENARIOS[escenario]
        informe = motor.evaluar_escenario(valores)

        st.markdown("**Proposiciones verdaderas:**")
        st.write([k for k, v in valores.items() if v])
        st.markdown("**Proposiciones falsas:**")
        st.write([k for k, v in valores.items() if not v])

        st.markdown("**Resultados de reglas:**")
        st.write(informe["resultados"])
        st.markdown("**Inferencias (Modus Ponens):**")
        st.write(informe["inferencias"])
        st.markdown("**Consistencia:**")
        st.write("Consistente" if informe["consistente"] else "Con violaciones")

    with st.expander("Razonamiento probabilistico"):
        st.subheader("Situaciones probabilisticas")
        sit_rows = [
            {
                "ID": s.id,
                "Descripcion": s.descripcion,
                "Dominio": s.dominio,
                "P base": s.p_base,
                "Umbral": s.umbral_accion,
            }
            for s in lprob.SITUACIONES
        ]
        st.table(sit_rows)

        st.subheader("Evaluacion de escenarios probabilisticos")
        motor_p = lprob.MotorProbabilistico()
        esc_p = st.selectbox("Escenario probabilistico", list(lprob.ESCENARIOS_PROB.keys()))
        evidencias = lprob.ESCENARIOS_PROB[esc_p]

        evaluaciones = []
        sp_map = {s.id: s for s in lprob.SITUACIONES}
        for sp_id, ev in evidencias.items():
            sp = sp_map[sp_id]
            p_aj = motor_p.calcular_p_ajustada(sp, ev)
            nivel, _ = motor_p.nivel_alerta(p_aj)
            evaluaciones.append(
                {
                    "ID": sp_id,
                    "Descripcion": sp.descripcion,
                    "P base": sp.p_base,
                    "P ajustada": p_aj,
                    "Nivel": nivel,
                }
            )

        decision = motor_p.decision_final(
            [
                {
                    "id": sp_id,
                    "p_ajustada": next(e["P ajustada"] for e in evaluaciones if e["ID"] == sp_id),
                    "umbral_accion": sp_map[sp_id].umbral_accion,
                }
                for sp_id in evidencias.keys()
            ]
        )

        st.table(evaluaciones)
        st.subheader("Decision global")
        st.write(decision)


def render_evidence():
    st.subheader("Evidencia de ejecucion")
    st.write("Resultados visibles para Partes A, B, J y K del entregable.")

    st.markdown("**Parte A — Simulacion de datos**")
    data_path = str(CSV_SENSORES)
    if CSV_SENSORES.exists():
        try:
            import pandas as pd

            df = pd.read_csv(data_path)
            st.write(f"Registros: {len(df)}")
            st.dataframe(df.head(15), use_container_width=True)
        except Exception as exc:
            st.warning(f"No se pudo cargar el CSV: {exc}")
    else:
        st.warning(f"No se encontro {CSV_SENSORES.name} en proyecto/data/.")

    st.markdown("**Parte B — Analisis de senales**")
    if os.path.exists(data_path):
        try:
            import pandas as pd

            df = pd.read_csv(data_path)
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df["vibracion"].tail(120), height=200)
                st.caption("Vibracion (Hz)")
            with col2:
                st.line_chart(df["temperatura"].tail(120), height=200)
                st.caption("Temperatura (C)")

            vib_anom = int((df["vibracion"] > 20).sum())
            temp_anom = int((df["temperatura"] > 30).sum())
            st.write(
                f"Anomalias detectadas: vibracion={vib_anom}, temperatura={temp_anom}."
            )
        except Exception as exc:
            st.warning(f"No se pudo analizar senales: {exc}")

    st.markdown("**Parte J — Programacion concurrente**")
    if st.button("Ejecutar concurrencia", key="btn_concurrent"):
        resultado = conc.ejecutar_concurrente(ciclos=3, incluir_vision=True)
        st.success("Concurrencia ejecutada.")
        st.write("Plan global:")
        st.json(resultado.plan_global)

    st.markdown("**Parte K — Optimizacion de despacho**")
    pedidos_demo = [
        {"id_pedido": "P001", "peso_kg": 50, "urgencia": 1, "tiempo_espera_minutos": 120},
        {"id_pedido": "P002", "peso_kg": 10, "urgencia": 3, "tiempo_espera_minutos": 15},
        {"id_pedido": "P003", "peso_kg": 30, "urgencia": 2, "tiempo_espera_minutos": 60},
        {"id_pedido": "P004", "peso_kg": 80, "urgencia": 1, "tiempo_espera_minutos": 200},
        {"id_pedido": "P005", "peso_kg": 15, "urgencia": 3, "tiempo_espera_minutos": 5},
    ]
    if st.button("Optimizar despacho", key="btn_opt"):
        seleccion = opt.optimizar_despachos(pedidos_demo, capacidad_maxima_vehiculo=100)
        st.write("Pedidos seleccionados:")
        st.write(seleccion)


def main():
    st.set_page_config(page_title="Sistema inteligente - Dashboard", layout="wide")
    inject_css()
    get_state()
    render_header()

    st.sidebar.markdown("### Navegacion")
    section = st.sidebar.radio(
        "Selecciona un modulo",
        [
            "Dashboard principal",
            "Panel de agentes",
            "Centro de alertas",
            "Graficos en tiempo real",
            "Panel de concurrencia",
            "Galeria de imagenes procesadas",
            "Laboratorio de razonamiento",
            "Evidencia de ejecucion",
        ],
    )

    if section == "Dashboard principal":
        render_dashboard()
    elif section == "Panel de agentes":
        render_agents()
    elif section == "Centro de alertas":
        render_alerts()
    elif section == "Graficos en tiempo real":
        render_realtime()
    elif section == "Panel de concurrencia":
        render_concurrency()
    elif section == "Galeria de imagenes procesadas":
        render_gallery()
    elif section == "Laboratorio de razonamiento":
        render_reasoning()
    elif section == "Evidencia de ejecucion":
        render_evidence()


if __name__ == "__main__":
    main()
