import os
import cv2
import numpy as np
import streamlit as st

import vision_cv
import logica_proposicional as lp
import logica_probabilidad as lprob
import sistema_experto as se
import agentes as ag


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


def main():
    # Configuracion general de la pagina.
    st.set_page_config(page_title="Sistema inteligente Almacen", layout="wide")
    st.title("Sistema inteligente multiagente - Almacen logistico")
    st.write(
        "Dashboard del sistema integrado. Cubre vision (Parte C), sistema experto (Parte D), "
        "representacion del conocimiento (Parte E), agentes (Parte F), logica proposicional "
        "(Parte G) y razonamiento probabilistico (Parte H)."
    )

    # Tabs principales para cada parte del proyecto.
    tab_c, tab_d, tab_e, tab_f, tab_g, tab_h = st.tabs([
        "Parte C - Vision",
        "Parte D - Sistema Experto",
        "Parte E - Conocimiento",
        "Parte F - Agentes",
        "Parte G - Logica",
        "Parte H - Probabilidad",
    ])

    with tab_c:
        st.header("Parte C - Procesamiento basico de imagenes")
        st.write(
            "Esta seccion procesa 3 imagenes .jpg ubicadas en la carpeta del proyecto. "
            "Se usan en orden alfabetico para: paquete danado, paquete en buen estado y zona obstruida."
        )

        # Detectar automaticamente las 3 imagenes .jpg del proyecto.
        jpgs = vision_cv.listar_jpgs()
        if len(jpgs) < 3:
            st.warning(
                f"Faltan imagenes para la Parte C. Se requieren 3 archivos .jpg en "
                f"`backend/ai_core/`. Encontradas: {len(jpgs)}. "
                f"Esta tab quedara inactiva, pero las otras pestanas (D, E, F, G, H) "
                f"funcionan con normalidad."
            )
            img_danado = img_bueno = img_obstruida = None
        else:
            # Asignar imagenes por orden alfabetico.
            img_danado, img_bueno, img_obstruida = jpgs[:3]

            st.markdown("**Imagenes detectadas (orden alfabetico):**")
            st.write(f"1) Paquete danado: {img_danado}")
            st.write(f"2) Paquete buen estado: {img_bueno}")
            st.write(f"3) Zona obstruida: {img_obstruida}")

        # Ejecutar el procesamiento cuando el usuario lo solicita.
        if img_danado and st.button("Procesar imagenes"):
            with st.spinner("Procesando..."):
                vision_cv.procesar_paquete_danado(img_danado)
                vision_cv.procesar_paquete_buen_estado(img_bueno)
                vision_cv.procesar_zona_obstruida(img_obstruida)

            st.success("Procesamiento completado. Resultados guardados en images_output.")

            st.header("Resultados")

            # Bloque 1: paquete danado (resultados y explicacion).
            col1, col2 = st.columns(2)
            with col1:
                show_image("Paquete danado - Original", os.path.join(vision_cv.RUTA_ENTRADA, img_danado))
                show_image("Paquete danado - Bordes (Canny)", os.path.join(vision_cv.RUTA_SALIDA, "img1_c2_canny.jpg"))
                show_image("Paquete danado - Umbral adaptativo", os.path.join(vision_cv.RUTA_SALIDA, "img1_c3_umbral_adaptativo.jpg"))
            with col2:
                st.markdown("**Tecnica aplicada:** Canny + umbral adaptativo.")
                st.write("**Observacion:** Bordes irregulares y zonas resaltadas indican deformaciones o manchas.")
                st.write("**Utilidad:** Detecta paquetes con dano probable para revision manual.")

            # Bloque 2: paquete en buen estado (resultados y explicacion).
            col3, col4 = st.columns(2)
            with col3:
                show_image("Paquete buen estado - Original", os.path.join(vision_cv.RUTA_ENTRADA, img_bueno))
                show_image("Paquete buen estado - Otsu", os.path.join(vision_cv.RUTA_SALIDA, "img2_c1_otsu.jpg"))
                show_image("Paquete buen estado - Apertura", os.path.join(vision_cv.RUTA_SALIDA, "img2_c2_apertura.jpg"))
            with col4:
                st.markdown("**Tecnica aplicada:** Otsu + apertura morfologica.")
                st.write("**Observacion:** Segmentacion compacta sugiere forma regular sin danios visibles.")
                st.write("**Utilidad:** Identifica paquetes aptos para despacho directo.")

            # Bloque 3: zona con obstruccion (resultados y explicacion).
            col5, col6 = st.columns(2)
            with col5:
                show_image("Zona obstruida - Original", os.path.join(vision_cv.RUTA_ENTRADA, img_obstruida))
                show_image("Zona obstruida - Canny", os.path.join(vision_cv.RUTA_SALIDA, "img3_c1_canny.jpg"))
                show_image("Zona obstruida - Mascara peligro", os.path.join(vision_cv.RUTA_SALIDA, "img3_c2_mascara_peligro.jpg"))
            with col6:
                st.markdown("**Tecnica aplicada:** Canny + segmentacion HSV.")
                st.write("**Observacion:** Zonas de color peligro y alta densidad de bordes indican bloqueo.")
                st.write("**Utilidad:** Permite alertar y bloquear rutas antes de accidentes.")

    # ─────────────────────────────────────────────────────────────────────
    # PARTE D — SISTEMA EXPERTO
    # ─────────────────────────────────────────────────────────────────────
    with tab_d:
        st.header("Parte D - Sistema experto basado en reglas")
        st.write(
            "Motor de inferencia hacia adelante (forward chaining) sobre 10 hechos y 8 reglas "
            "if/then. Selecciona un escenario o construye uno manualmente y observa las "
            "conclusiones derivadas, sus prioridades y la cadena de razonamiento."
        )

        modo_d = st.radio(
            "Modo de uso",
            ["Escenario predefinido", "Selección manual de hechos"],
            horizontal=True,
        )

        if modo_d == "Escenario predefinido":
            nombre_esc = st.selectbox(
                "Escenario", list(se.ESCENARIOS_EXPERTO.keys())
            )
            hechos_activar = se.ESCENARIOS_EXPERTO[nombre_esc]
        else:
            hechos_activar = st.multiselect(
                "Hechos a asertar",
                options=list(se.HECHOS_BASE.keys()),
                default=["vibracion_anomala", "mantenimiento_diferido"],
            )

        if st.button("Ejecutar motor de inferencia", key="btn_exp"):
            resultado = se.ejecutar_escenario(hechos_activar)

            st.markdown("**Hechos iniciales:**")
            st.write(resultado["hechos_iniciales"])

            st.markdown(f"**Conclusiones derivadas: {resultado['n_conclusiones']}**")
            if resultado["conclusiones"]:
                filas_conc = [{
                    "Regla": c["regla_id"],
                    "Conclusión": c["hecho_nuevo"],
                    "Prioridad": c["prioridad"],
                    "Confianza": f"{c['confianza']:.0%}",
                    "Acción recomendada": c["accion"],
                } for c in resultado["conclusiones"]]
                st.table(filas_conc)

                prio_max = resultado["prioridad_max"]
                if prio_max == 4:
                    st.error("Prioridad CRÍTICA — se requiere parada de emergencia.")
                elif prio_max == 3:
                    st.warning("Prioridad ALTA — supervisión activa requerida.")
                else:
                    st.info("Prioridad media o baja — monitoreo de rutina.")
            else:
                st.success("Sin conclusiones — operación normal.")

            with st.expander("Traza de inferencia (cómo razonó el motor)"):
                for linea in resultado["traza"]:
                    st.text(linea)

    # ─────────────────────────────────────────────────────────────────────
    # PARTE E — REPRESENTACIÓN DEL CONOCIMIENTO
    # ─────────────────────────────────────────────────────────────────────
    with tab_e:
        st.header("Parte E - Representación del conocimiento")
        st.write(
            "Visualización de la base de hechos, reglas y la tabla de conocimiento. "
            "Esta sección documenta CÓMO se modelan internamente los conceptos del almacén."
        )

        st.subheader("Base de hechos (10)")
        filas_h = [{
            "Hecho": h.nombre,
            "Dominio": h.dominio,
            "Descripción": h.descripcion,
        } for h in se.HECHOS_BASE.values()]
        st.table(filas_h)

        st.subheader("Base de reglas (8)")
        filas_r = [{
            "ID": r.id,
            "Antecedentes": " ∧ ".join(r.antecedentes),
            "Consecuente": r.consecuente,
            "Prioridad": r.prioridad,
            "Acción": r.accion,
        } for r in se.REGLAS]
        st.table(filas_r)

        st.subheader("Tabla de conocimiento (hechos ↔ reglas)")
        st.write(
            "Cada fila indica en qué reglas participa el hecho y si es un "
            "hecho primitivo (introducido por sensor) o derivado (concluido por el motor)."
        )
        tabla = se.construir_tabla_conocimiento()
        filas_t = [{
            "Hecho": t["hecho"],
            "Dominio": t["dominio"],
            "Tipo": "derivado" if t["es_conclusion"] else "primitivo",
            "Reglas que lo usan": ", ".join(t["reglas_uso"]) if t["reglas_uso"] else "—",
        } for t in tabla]
        st.table(filas_t)

        st.subheader("Relaciones causales")
        st.write(
            "Cadenas típicas de razonamiento: un hecho primitivo puede gatillar "
            "varias conclusiones encadenadas."
        )
        filas_rel = [{
            "Hecho origen": k,
            "Conclusiones posibles": " → ".join(v),
        } for k, v in se.RELACIONES_CAUSALES.items()]
        st.table(filas_rel)

    # ─────────────────────────────────────────────────────────────────────
    # PARTE F — AGENTES BASADOS EN CONOCIMIENTO
    # ─────────────────────────────────────────────────────────────────────
    with tab_f:
        st.header("Parte F - Agentes basados en conocimiento")
        st.write(
            "Cinco agentes especializados (Sensor, AnalizadorSeñales, AnalizadorImágenes, "
            "Decisor, Coordinador) que se comunican mediante mensajes tipados. "
            "Aquí puedes ejecutar el pipeline multiagente completo en modo secuencial."
        )

        st.subheader("Agentes del sistema")
        info_agentes = [
            {
                "Agente": "AgenteSensor",
                "Conocimiento": "umbrales nominales + sensores disponibles",
                "Entrada": "—",
                "Salida": "Mensaje 'lectura' con variables ambientales",
                "Cómo decide": "muestrea CSV o simula lectura",
            },
            {
                "Agente": "AgenteAnalizadorSenales",
                "Conocimiento": "umbrales + ventana móvil de 60 lecturas",
                "Entrada": "Mensaje 'lectura'",
                "Salida": "Mensaje 'analisis' con hechos lógicos",
                "Cómo decide": "umbralización + conversión a hechos",
            },
            {
                "Agente": "AgenteAnalizadorImagenes",
                "Conocimiento": "reporte JSON de vision_cv",
                "Entrada": "Mensaje 'comando' o evento periódico",
                "Salida": "Mensaje 'analisis' con hechos de visión",
                "Cómo decide": "traduce niveles de alerta visual a hechos",
            },
            {
                "Agente": "AgenteDecisor",
                "Conocimiento": "10 hechos + 8 reglas (sistema experto)",
                "Entrada": "Mensaje 'analisis' de los analizadores",
                "Salida": "Mensaje 'decision' con conclusiones priorizadas",
                "Cómo decide": "forward chaining sobre reglas if/then",
            },
            {
                "Agente": "AgenteCoordinador",
                "Conocimiento": "política global + motor probabilístico (Parte H)",
                "Entrada": "Mensajes 'decision' del Decisor",
                "Salida": "Mensaje 'alerta' con plan global",
                "Cómo decide": "fusión Noisy-OR + clasificación por nivel",
            },
        ]
        st.table(info_agentes)

        st.subheader("Ejecución del pipeline multiagente")
        col_a, col_b = st.columns(2)
        with col_a:
            ciclos_n = st.slider("Número de ciclos a ejecutar", 1, 10, 3)
        with col_b:
            usar_vision = st.checkbox("Incluir agente de visión", value=True)

        if st.button("Ejecutar pipeline multiagente", key="btn_agt"):
            csv_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "simulacion_almacen.csv"
            )
            csv_path = csv_path if os.path.exists(csv_path) else None

            sistema = ag.SistemaMultiAgente(csv_sensores=csv_path)
            resultados_ciclos = []
            for i in range(ciclos_n):
                r = sistema.ciclo(incluir_vision=usar_vision)
                resultados_ciclos.append(r)

            st.success(f"Pipeline ejecutado: {ciclos_n} ciclos, "
                       f"{len(sistema.historial)} mensajes intercambiados.")

            st.markdown("**Plan global del último ciclo:**")
            ultimo = resultados_ciclos[-1]["plan_global"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Nivel de riesgo", ultimo.get("nivel_global", "—"))
            col2.metric("Riesgo global", f"{ultimo.get('riesgo_global_pct', 0)}%")
            col3.metric("Mensajes", len(sistema.historial))

            st.markdown(f"**Directiva:** {ultimo.get('directiva_principal', '—')}")

            acciones = ultimo.get("acciones_priorizadas", [])
            if acciones:
                st.markdown("**Acciones priorizadas:**")
                st.table([{
                    "Orden"    : a["orden"],
                    "Prioridad": a["prioridad"],
                    "Regla"    : a["regla"],
                    "Conclusión": a["conclusion"],
                    "Acción"   : a["accion"],
                } for a in acciones])
            else:
                st.info("Sin acciones requeridas — operación normal.")

            with st.expander("Estado de los agentes"):
                st.table(sistema.resumen_estado())

            with st.expander("Historial de mensajes intercambiados"):
                st.table([{
                    "id"      : m.id_mensaje,
                    "origen"  : m.origen,
                    "destino" : m.destino,
                    "tipo"    : m.tipo,
                    "prioridad": m.prioridad,
                } for m in sistema.historial])

    with tab_g:
        st.header("Parte G - Logica general y proposicional")
        st.write("Se muestran proposiciones, reglas y evaluacion de escenarios.")

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
        escenario = st.selectbox("Selecciona un escenario", list(lp.ESCENARIOS.keys()))
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

    with tab_h:
        st.header("Parte H - Razonamiento probabilistico")
        st.write("Se muestran situaciones probabilisticas y evaluacion de escenarios.")

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
        esc_p = st.selectbox("Selecciona un escenario probabilistico", list(lprob.ESCENARIOS_PROB.keys()))
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


if __name__ == "__main__":
    main()
