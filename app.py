# -*- coding: utf-8 -*-
"""
app.py
Interfaz Streamlit para el modelo de bagginess en testliner.
Mantiene la lógica original del notebook de Colab (sin reescribirla).
"""

import io
import os
import json
import pandas as pd
import streamlit as st

from modules.data_processing import (
    FEATURES, TARGET,
    cargar_excel, preparar_dataset, parsear_durezas, calcular_variables,
    variables_a_dataframe,
)
from modules.model_training import (
    split_datos, entrenar_modelo, evaluar_modelo,
    guardar_modelo, cargar_modelo, MODEL_PATH_DEFAULT,
)
from modules.prediction import predecir_rollo
from modules.visualizations import (
    scatter_prob_vs_var, plot_matriz_confusion, plot_importancias,
    plot_distribucion_variable, plot_perfil_rollo,
)
from modules.audit import registrar_evento, leer_eventos


# =========================
# Configuración visual
# =========================
st.set_page_config(
    page_title="Predicción Bagginess | Testliner",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
:root { color-scheme: light; }
.main { background-color: #fafafa; }
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { color: #1f2d3d; }
.metric-card {
  background:#ffffff; border:1px solid #e5e7eb; border-radius:10px;
  padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,0.04);
}
.stButton>button {
  background:#1f4e79; color:white; border:0; border-radius:6px;
  padding:8px 18px; font-weight:500;
}
.stButton>button:hover { background:#163a5c; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Predicción de Bagginess — Testliner")
st.caption("Interfaz industrial basada en el notebook original de Colab. Lógica de cálculo y modelo sin modificaciones.")


# =========================
# Session state
# =========================
if "df" not in st.session_state: st.session_state.df = None
if "modelo" not in st.session_state: st.session_state.modelo = None
if "eval" not in st.session_state: st.session_state.eval = None
if "archivo_nombre" not in st.session_state: st.session_state.archivo_nombre = None


# =========================
# Sidebar — navegación
# =========================
seccion = st.sidebar.radio(
    "Secciones",
    ["1. Carga de datos",
     "2. Entrenamiento",
     "3. Resultados",
     "4. Predicción de rollo",
     "5. Auditoría",
     "6. Comparación con Colab"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo actual:**")
if os.path.exists(MODEL_PATH_DEFAULT):
    st.sidebar.success(f"Guardado: {os.path.basename(MODEL_PATH_DEFAULT)}")
    if st.sidebar.button("Cargar modelo guardado"):
        try:
            st.session_state.modelo = cargar_modelo(MODEL_PATH_DEFAULT)
            st.sidebar.success("Modelo cargado en memoria.")
            registrar_evento("modelo_cargado", {"path": MODEL_PATH_DEFAULT})
        except Exception as e:
            st.sidebar.error(f"Error al cargar: {e}")
else:
    st.sidebar.info("Aún no hay modelo entrenado.")


# =========================
# 1. CARGA DE DATOS
# =========================
if seccion.startswith("1"):
    st.header("1️⃣ Carga de datos")

    archivo = st.file_uploader("Subir archivo Excel (.xlsx)", type=["xlsx"])
    hoja = st.text_input("Nombre de la hoja", value="Dureza")

    if archivo is not None:
        try:
            df = cargar_excel(archivo, sheet_name=hoja)
            st.session_state.df = df
            st.session_state.archivo_nombre = archivo.name

            faltantes = [c for c in FEATURES + [TARGET] if c not in df.columns]
            if faltantes:
                st.error(f"Faltan columnas requeridas: {faltantes}")
            else:
                st.success(f"Archivo válido: {archivo.name} — {df.shape[0]} filas, {df.shape[1]} columnas")

            st.subheader("Vista previa")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Estadísticas descriptivas")
            st.dataframe(df[FEATURES].describe(), use_container_width=True)

            registrar_evento("archivo_cargado", {
                "nombre": archivo.name, "filas": int(df.shape[0]),
                "columnas": list(df.columns), "hoja": hoja,
            })
        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")


# =========================
# 2. ENTRENAMIENTO
# =========================
elif seccion.startswith("2"):
    st.header("2️⃣ Entrenamiento del modelo")

    if st.session_state.df is None:
        st.warning("Primero carga un archivo en la sección 1.")
    else:
        df = st.session_state.df
        st.write(f"Dataset actual: **{st.session_state.archivo_nombre}** — {df.shape[0]} filas")

        if st.button("🚀 Entrenar modelo"):
            try:
                with st.status("Entrenando XGBoost...", expanded=True) as status:
                    st.write("Preparando dataset...")
                    X, y, df_model = preparar_dataset(df)
                    st.write(f"Filas utilizables: {len(df_model)}")
                    st.write("Distribución de clases:")
                    st.write(y.value_counts().to_dict())

                    st.write("Split train/test 75/25 estratificado...")
                    X_train, X_test, y_train, y_test = split_datos(X, y)

                    st.write("Ajustando XGBoost (n_estimators=400, max_depth=4, lr=0.05)...")
                    xgb = entrenar_modelo(X_train, y_train)

                    st.write("Evaluando con reglas híbridas (modelo + físicas)...")
                    resultado = evaluar_modelo(xgb, X, y, df_model)

                    st.write("Guardando modelo...")
                    path = guardar_modelo(xgb)

                    st.session_state.modelo = xgb
                    st.session_state.eval = resultado
                    status.update(label="✅ Entrenamiento completado", state="complete")

                registrar_evento("modelo_entrenado", {
                    "n_filas": int(len(df_model)),
                    "distribucion": y.value_counts().to_dict(),
                    "prob_stats": resultado["prob_stats"],
                    "modelo_path": path,
                })
                st.success(f"Modelo guardado en: {path}")
            except Exception as e:
                st.error(f"Error en entrenamiento: {e}")


# =========================
# 3. RESULTADOS
# =========================
elif seccion.startswith("3"):
    st.header("3️⃣ Resultados y métricas")

    if st.session_state.eval is None:
        st.warning("Entrena el modelo primero (sección 2).")
    else:
        r = st.session_state.eval
        df_scores = r["df_scores"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Prob. mínima", f"{r['prob_stats']['min']:.3f}")
        c2.metric("Prob. máxima", f"{r['prob_stats']['max']:.3f}")
        c3.metric("Desv. probabilidad", f"{r['prob_stats']['std']:.3f}")

        st.subheader("Matriz de confusión")
        st.pyplot(plot_matriz_confusion(r["confusion_matrix"]))

        st.subheader("Reporte de clasificación")
        st.code(r["classification_report_text"])

        st.subheader("Importancia de variables")
        col_a, col_b = st.columns([1, 1])
        col_a.pyplot(plot_importancias(r["importancias"]))
        col_b.dataframe(r["importancias"], use_container_width=True)

        st.subheader("Distribución de decisiones")
        st.write(r["distribucion_decisiones"])

        st.subheader("Probabilidad vs variables clave")
        tabs = st.tabs(["Variabilidad", "Simetría", "Curvatura"])
        with tabs[0]:
            st.pyplot(scatter_prob_vs_var(df_scores, "Variabilidad"))
        with tabs[1]:
            st.pyplot(scatter_prob_vs_var(df_scores, "Simetría"))
        with tabs[2]:
            st.pyplot(scatter_prob_vs_var(df_scores, "Curvatura"))

        st.subheader("Distribuciones de variables")
        var_dist = st.selectbox("Variable", FEATURES, index=3)
        st.pyplot(plot_distribucion_variable(df_scores, var_dist))

        st.subheader("Descargar resultados")
        csv = df_scores.to_csv(index=False).encode("utf-8")
        st.download_button("📥 df_scores.csv", csv, "df_scores.csv", "text/csv")


# =========================
# 4. PREDICCIÓN DE ROLLO
# =========================
elif seccion.startswith("4"):
    st.header("4️⃣ Predicción de rollo individual")

    if st.session_state.modelo is None:
        st.warning("Carga o entrena un modelo primero.")
    else:
        xgb = st.session_state.modelo
        st.write("Ingresa las **20 mediciones de dureza** a lo largo del ancho del rollo.")

        modo = st.radio("Modo de ingreso", ["Formulario (20 campos)", "Pegar texto"], horizontal=True)

        arr = None
        if modo.startswith("Formulario"):
            cols = st.columns(5)
            valores = []
            for i in range(20):
                v = cols[i % 5].number_input(f"P{i+1}", value=0.0, format="%.3f", key=f"p{i}")
                valores.append(v)
            if st.button("🔮 Predecir"):
                try:
                    arr = parsear_durezas(" ".join(str(v) for v in valores))
                except Exception as e:
                    st.error(str(e))
        else:
            entrada = st.text_area("Pega los 20 valores (separados por espacio, salto de línea, ; o coma decimal):", height=150)
            if st.button("🔮 Predecir"):
                try:
                    arr = parsear_durezas(entrada)
                except Exception as e:
                    st.error(str(e))

        if arr is not None:
            res = predecir_rollo(xgb, arr)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Resultado")
                color = {"🔴 RECHAZAR": "#c0392b", "🟡 ALERTA": "#d4ac0d", "🟢 OK": "#27ae60"}[res["decision"]]
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<h3 style='color:{color};margin:0'>{res['decision']}</h3>"
                    f"<p style='margin:4px 0 0 0'>Probabilidad de reclamo: <b>{res['probabilidad']:.4f}</b></p>"
                    f"</div>", unsafe_allow_html=True,
                )
                st.subheader("Variables calculadas")
                st.dataframe(pd.DataFrame([res["variables"]]).T.rename(columns={0: "Valor"}))

            with col2:
                st.pyplot(plot_perfil_rollo(res["durezas"]))

            registrar_evento("prediccion_rollo", {
                "durezas": res["durezas"],
                "variables": res["variables"],
                "probabilidad": res["probabilidad"],
                "decision": res["decision"],
            })

            st.download_button(
                "📥 Descargar resultado JSON",
                json.dumps({
                    "durezas": res["durezas"],
                    "variables": res["variables"],
                    "probabilidad": res["probabilidad"],
                    "decision": res["decision"],
                }, ensure_ascii=False, indent=2).encode("utf-8"),
                "prediccion_rollo.json", "application/json",
            )


# =========================
# 5. AUDITORÍA
# =========================
elif seccion.startswith("5"):
    st.header("5️⃣ Auditoría")

    eventos = leer_eventos(limit=500)
    st.write(f"Total eventos registrados (últimos 500): **{len(eventos)}**")

    if eventos:
        tipos = sorted({e["tipo"] for e in eventos})
        filtro = st.multiselect("Filtrar por tipo", tipos, default=tipos)
        vista = [e for e in eventos if e["tipo"] in filtro]
        st.dataframe(
            pd.DataFrame([{
                "timestamp": e["timestamp"], "tipo": e["tipo"],
                "detalle": json.dumps(e["payload"], ensure_ascii=False)[:300]
            } for e in vista]),
            use_container_width=True,
        )

        st.subheader("Detalle de evento")
        idx = st.number_input("Índice", 0, max(len(vista) - 1, 0), 0)
        if vista:
            st.json(vista[idx])

        blob = "\n".join(json.dumps(e, ensure_ascii=False) for e in vista).encode("utf-8")
        st.download_button("📥 Descargar log (JSONL)", blob, "auditoria.jsonl", "application/jsonl")
    else:
        st.info("Aún no hay eventos registrados.")


# =========================
# 6. COMPARACIÓN CON COLAB
# =========================
else:
    st.header("6️⃣ Comparación con Colab")
    st.write("Sube un archivo con los resultados obtenidos en Colab para compararlos con esta ejecución.")

    archivo_colab = st.file_uploader("Resultados de Colab (CSV con columnas: Prob_modelo, Decision, Pred_final)", type=["csv"])

    if archivo_colab and st.session_state.eval is not None:
        try:
            df_colab = pd.read_csv(archivo_colab)
            df_app = st.session_state.eval["df_scores"].reset_index(drop=True)
            n = min(len(df_app), len(df_colab))
            comp = pd.DataFrame({
                "Prob_app": df_app["Prob_modelo"].iloc[:n].values,
                "Prob_colab": df_colab["Prob_modelo"].iloc[:n].values,
            })
            comp["Diff_prob"] = (comp["Prob_app"] - comp["Prob_colab"]).abs()
            if "Decision" in df_colab.columns:
                comp["Dec_app"] = df_app["Decision"].iloc[:n].values
                comp["Dec_colab"] = df_colab["Decision"].iloc[:n].values
                comp["Coincide"] = comp["Dec_app"] == comp["Dec_colab"]

            st.subheader("Resumen")
            c1, c2, c3 = st.columns(3)
            c1.metric("Filas comparadas", n)
            c2.metric("Diff prob. máx.", f"{comp['Diff_prob'].max():.5f}")
            if "Coincide" in comp.columns:
                c3.metric("% decisiones idénticas", f"{100*comp['Coincide'].mean():.2f}%")

            st.dataframe(comp, use_container_width=True)
            st.download_button("📥 Descargar comparación",
                               comp.to_csv(index=False).encode("utf-8"),
                               "comparacion_colab.csv", "text/csv")

            registrar_evento("comparacion_colab", {
                "filas": int(n),
                "diff_prob_max": float(comp["Diff_prob"].max()),
                "coincidencia_pct": float(100 * comp["Coincide"].mean()) if "Coincide" in comp.columns else None,
            })
        except Exception as e:
            st.error(f"Error comparando: {e}")
    elif archivo_colab and st.session_state.eval is None:
        st.warning("Entrena el modelo primero para comparar.")
