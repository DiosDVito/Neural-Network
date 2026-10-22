"""
Daniel Esparza Arizpe - A01637076

App Streamlit: clasifica puntos GPS de Monterrey vs Lille.

Basado en:
  https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/
  https://github.com/adsoftsito/tecnologias-construccion/blob/main/neural_network_numpy.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from datos import DATASET_ID, LILLE, MONTERREY, generar_datasets, normalizar
from mapas import dataframe_mapa, mostrar_mapa
from modelo import crear_red, entrenamiento, mse, prediccion
from validacion import correr_validacion, guardar_validacion, mostrar_validacion

st.set_page_config(page_title="Red neuronal de ciudades", layout="wide")
st.title("Clasificador de ciudades")
st.header("Monterrey vs Lille")
st.caption(
    "Red neuronal programada desde 0 en NumPy. "
    "Puntos leídos de monterrey.csv y lille.csv."
)

with st.sidebar:
    st.subheader("Entrenamiento")
    epochs = st.slider("Epochs", 100, 2000, 1000, 100)
    lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.5, value=0.01, format="%.4f")
    seed = st.number_input("Semilla", min_value=0, value=42, step=1)
    entrenar = st.button("Entrenar red", type="primary")

if (
    "X" not in st.session_state
    or st.session_state.get("seed") != seed
    or st.session_state.get("dataset_id") != DATASET_ID
):
    X, Y = generar_datasets(seed=int(seed))
    X_norm, media, std = normalizar(X)
    st.session_state.X = X
    st.session_state.Y = Y
    st.session_state.X_norm = X_norm
    st.session_state.media = media
    st.session_state.std = std
    st.session_state.seed = int(seed)
    st.session_state.dataset_id = DATASET_ID
    st.session_state.red = None
    st.session_state.error = []
    st.session_state.preds = None

X = st.session_state.X
Y = st.session_state.Y
df_map = dataframe_mapa(X, Y)

st.subheader("Datasets GPS")
m1, m2, m3 = st.columns(3)
m1.metric("Puntos Monterrey (clase 0)", int((Y == 0).sum()))
m2.metric("Puntos Lille (clase 1)", int((Y == 1).sum()))
m3.write("Azul = Monterrey · Rojo = Lille")

col_mty, col_lille = st.columns(2)
with col_mty:
    st.markdown("**Monterrey**")
    mostrar_mapa(df_map[df_map["clase"] == "Monterrey"], MONTERREY, zoom=12)
with col_lille:
    st.markdown("**Lille**")
    mostrar_mapa(df_map[df_map["clase"] == "Lille"], LILLE, zoom=13)

st.dataframe(df_map[["lat", "lon", "clase"]].head(8), width="stretch")

if entrenar:
    red = crear_red()
    historial = []
    ultima = None
    barra = st.progress(0, text="Entrenando...")
    for epoch in range(epochs):
        ultima = entrenamiento(
            st.session_state.X_norm, st.session_state.Y, red, lr=float(lr)
        )
        historial.append(mse(np.round(ultima), st.session_state.Y)[0])
        if epoch % max(epochs // 20, 1) == 0 or epoch == epochs - 1:
            barra.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs}")
    barra.empty()
    st.session_state.red = red
    st.session_state.error = historial
    st.session_state.preds = ultima

if st.session_state.red is not None:
    preds = st.session_state.preds
    pred_clase = (preds >= 0.5).astype(int)
    accuracy = float(np.mean(pred_clase == st.session_state.Y))

    st.subheader("Resultado del entrenamiento")
    m1, m2, m3 = st.columns(3)
    m1.metric("MSE final", f"{st.session_state.error[-1]:.4f}")
    m2.metric("Accuracy", f"{accuracy:.1%}")
    m3.metric("Epochs", len(st.session_state.error))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(st.session_state.error, color="#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Error durante el entrenamiento")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Predecir una ciudad")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        lat = st.number_input("Latitud", min_value=-90.0, max_value=90.0, value=MONTERREY[0], format="%.4f")
    with c2:
        lon = st.number_input("Longitud", min_value=-180.0, max_value=180.0, value=MONTERREY[1], format="%.4f")
    with c3:
        st.write("")
        st.write("")
        predecir = st.button("Predecir")

    if predecir:
        punto = np.array([[lat, lon]])
        punto_norm, _, _ = normalizar(
            punto, st.session_state.media, st.session_state.std
        )
        prob = float(prediccion(punto_norm, st.session_state.red)[0, 0])
        etiqueta = "Lille" if prob >= 0.5 else "Monterrey"
        st.success(f"**{etiqueta}** · probabilidad clase 1 = {prob:.3f}")
        mostrar_mapa(
            pd.DataFrame([{"lat": lat, "lon": lon, "clase": etiqueta, "color": "#2ca02c"}]),
            centro=(lat, lon),
            zoom=12,
            height=360,
        )
else:
    st.info("Usa el botón **Entrenar red** en la barra lateral para ajustar pesos y bias.")

st.divider()
st.subheader("Validación: épocas 0 y 1")
st.caption(
    "Toma 20 puntos GPS (10 Monterrey + 10 Lille) y ejecuta solo las épocas 0 y 1. "
    "Muestra X, W, b, z, a y delta por capa, como en el código de referencia."
)
if st.button("Ejecutar validación"):
    resultado = correr_validacion(
        st.session_state.X,
        st.session_state.Y,
        st.session_state.X_norm,
        lr=float(lr),
        seed=int(seed),
    )
    ruta, texto = guardar_validacion(resultado)
    st.session_state.validacion = resultado
    st.session_state.validacion_ruta = str(ruta)
    st.session_state.validacion_texto = texto

if st.session_state.get("validacion") is not None:
    st.success(f"Guardado en `{st.session_state.validacion_ruta}`")
    st.download_button(
        "Descargar validacion_epocas_0_1.txt",
        data=st.session_state.validacion_texto,
        file_name="validacion_epocas_0_1.txt",
        mime="text/plain",
    )
    mostrar_validacion(st.session_state.validacion)
