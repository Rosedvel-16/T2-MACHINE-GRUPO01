import os
import cv2
import numpy as np
import streamlit as st

import vision_cv
import logica_proposicional as lp
import logica_probabilidad as lprob


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
    st.set_page_config(page_title="Sistema inteligente", layout="wide")
    st.title("Sistema inteligente - Partes C, G y H")
    st.write("App unificada para vision, logica proposicional y razonamiento probabilistico.")

    # Tabs principales para cada parte del proyecto.
    tab_c, tab_g, tab_h = st.tabs(["Parte C - Vision", "Parte G - Logica", "Parte H - Probabilidad"])

    with tab_c:
        st.header("Parte C - Procesamiento basico de imagenes")
        st.write(
            "Esta seccion procesa 3 imagenes .jpg ubicadas en la carpeta del proyecto. "
            "Se usan en orden alfabetico para: paquete danado, paquete en buen estado y zona obstruida."
        )

        # Detectar automaticamente las 3 imagenes .jpg del proyecto.
        jpgs = vision_cv.listar_jpgs()
        if len(jpgs) < 3:
            st.error(
                f"Se requieren 3 imagenes .jpg en la carpeta del proyecto. Encontradas: {len(jpgs)}"
            )
            st.stop()

        # Asignar imagenes por orden alfabetico.
        img_danado, img_bueno, img_obstruida = jpgs[:3]

        st.markdown("**Imagenes detectadas (orden alfabetico):**")
        st.write(f"1) Paquete danado: {img_danado}")
        st.write(f"2) Paquete buen estado: {img_bueno}")
        st.write(f"3) Zona obstruida: {img_obstruida}")

        # Ejecutar el procesamiento cuando el usuario lo solicita.
        if st.button("Procesar imagenes"):
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
