"""
Daniel Esparza Arizpe - A01637076

Basado en:
  https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/
  https://github.com/adsoftsito/tecnologias-construccion/blob/main/neural_network_numpy.py
"""

from __future__ import annotations

import math
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

sigmoid = (
    lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    lambda x: x * (1 - x),
)


def derivada_relu(x):
    d = np.zeros_like(x)
    d[x > 0] = 1
    return d


relu = (
    lambda x: x * (x > 0),
    lambda x: derivada_relu(x),
)


class capa:
    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.round(
            stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas).reshape(
                1, n_neuronas
            ),
            3,
        )
        self.W = np.round(
            stats.truncnorm.rvs(
                -1, 1, loc=0, scale=1, size=n_neuronas * n_neuronas_capa_anterior
            ).reshape(n_neuronas_capa_anterior, n_neuronas),
            3,
        )


def mse(Ypredich, Yreal):
    diff = np.array(Ypredich) - np.array(Yreal)
    return (np.mean(diff**2), diff)


def prediccion(X, red_neuronal):
    output = X
    for layer in red_neuronal:
        z = output @ layer.W + layer.b
        output = layer.funcion_act[0](z)
    return output


def entrenamiento(X, Y, red_neuronal, lr=0.01):
    output = [X]
    for num_capa in range(len(red_neuronal)):
        z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
        a = red_neuronal[num_capa].funcion_act[0](z)
        output.append(a)

    back = list(range(len(output) - 1))
    back.reverse()
    delta = []

    for idx in back:
        a = output[idx + 1]
        if idx == back[0]:
            x = mse(a, Y)[1] * red_neuronal[idx].funcion_act[1](a)
            delta.append(x)
        else:
            x = delta[-1] @ W_temp * red_neuronal[idx].funcion_act[1](a)
            delta.append(x)

        W_temp = red_neuronal[idx].W.transpose()
        red_neuronal[idx].b = (
            red_neuronal[idx].b - np.mean(delta[-1], axis=0, keepdims=True) * lr
        )
        red_neuronal[idx].W = (
            red_neuronal[idx].W - output[idx].transpose() @ delta[-1] * lr
        )

    return output[-1]


def crear_red():
    neuronas = [2, 4, 8, 1]
    funciones_activacion = [relu, relu, sigmoid]
    red = []
    for paso in range(len(neuronas) - 1):
        red.append(capa(neuronas[paso], neuronas[paso + 1], funciones_activacion[paso]))
    return red


NOMBRES_CAPA = ["Capa 1 (2→4, ReLU)", "Capa 2 (4→8, ReLU)", "Capa 3 (8→1, Sigmoide)"]


def seleccionar_20_puntos(X, Y, X_norm):
    idx0 = np.where(Y.ravel() == 0)[0][:10]
    idx1 = np.where(Y.ravel() == 1)[0][:10]
    idx = np.concatenate([idx0, idx1])
    return X[idx], Y[idx], X_norm[idx]


def entrenamiento_traza(X, Y, red_neuronal, lr=0.01):
    """Un paso de entrenamiento, guardando X, W, b, z, a y delta por capa."""
    output = [X]
    capas = []
    for num_capa in range(len(red_neuronal)):
        W = red_neuronal[num_capa].W.copy()
        b = red_neuronal[num_capa].b.copy()
        z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
        a = red_neuronal[num_capa].funcion_act[0](z)
        output.append(a)
        capas.append({"W": W, "b": b, "z": z.copy(), "a": a.copy()})

    back = list(range(len(output) - 1))
    back.reverse()
    delta = []
    deltas = [None] * len(red_neuronal)

    for idx in back:
        a = output[idx + 1]
        if idx == back[0]:
            x = mse(a, Y)[1] * red_neuronal[idx].funcion_act[1](a)
        else:
            x = delta[-1] @ W_temp * red_neuronal[idx].funcion_act[1](a)
        delta.append(x)
        deltas[idx] = x.copy()
        W_temp = red_neuronal[idx].W.transpose()
        red_neuronal[idx].b = (
            red_neuronal[idx].b - np.mean(delta[-1], axis=0, keepdims=True) * lr
        )
        red_neuronal[idx].W = (
            red_neuronal[idx].W - output[idx].transpose() @ delta[-1] * lr
        )

    for i, snap in enumerate(capas):
        snap["delta"] = deltas[i]

    return {"X": X.copy(), "Y": Y.copy(), "capas": capas}


def correr_validacion(X_gps, Y, X_norm, lr=0.01, seed=42):
    X20, Y20, Xn20 = seleccionar_20_puntos(X_gps, Y, X_norm)
    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    red = crear_red()
    np.random.set_state(rng_state)

    epocas = []
    for epoch in (0, 1):
        traza = entrenamiento_traza(Xn20, Y20, red, lr=float(lr))
        traza["epoch"] = epoch
        epocas.append(traza)

    return {"X_gps": X20, "Y": Y20, "X": Xn20, "epocas": epocas}


RUTA_VALIDACION = Path(__file__).resolve().parent / "validacion_epocas_0_1.txt"


def _bloque_matriz(nombre, arr):
    data = np.round(np.array(arr), 4)
    return f"{nombre}  shape={data.shape}\n{np.array2string(data, precision=4, suppress_small=True)}\n"


def reporte_validacion(resultado):
    lineas = [
        "Validación red neuronal — épocas 0 y 1",
        "20 puntos GPS (10 Monterrey + 10 Lille)",
        "",
        _bloque_matriz("X_gps (lat, lon)", resultado["X_gps"]),
        _bloque_matriz("Y", resultado["Y"]),
        _bloque_matriz("X (normalizado)", resultado["X"]),
    ]
    for traza in resultado["epocas"]:
        lineas.append("=" * 72)
        lineas.append(f"ÉPOCA {traza['epoch']}")
        lineas.append("=" * 72)
        lineas.append(_bloque_matriz("X", traza["X"]))
        lineas.append(_bloque_matriz("Y", traza["Y"]))
        for i, capa_snap in enumerate(traza["capas"]):
            lineas.append("-" * 72)
            lineas.append(NOMBRES_CAPA[i])
            lineas.append("-" * 72)
            for clave in ("W", "b", "z", "a", "delta"):
                lineas.append(_bloque_matriz(clave, capa_snap[clave]))
    return "\n".join(lineas)


def guardar_validacion(resultado, ruta=RUTA_VALIDACION):
    texto = reporte_validacion(resultado)
    ruta.write_text(texto, encoding="utf-8")
    return ruta, texto


def df_matriz(arr, prefijo="c"):
    data = np.round(np.array(arr), 4)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    cols = [f"{prefijo}{i}" for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=cols)


def mostrar_validacion(resultado):
    st.markdown("**20 puntos GPS seleccionados** (10 Monterrey + 10 Lille)")
    puntos = pd.DataFrame(resultado["X_gps"], columns=["lat", "lon"])
    puntos["Y"] = resultado["Y"].ravel()
    puntos["clase"] = np.where(puntos["Y"] == 0, "Monterrey", "Lille")
    st.dataframe(puntos, width="stretch")

    st.markdown("**X de entrada a la red** (lat/lon normalizados, 20×2)")
    st.dataframe(df_matriz(resultado["X"], prefijo="x"), width="stretch")

    for traza in resultado["epocas"]:
        with st.expander(f"Época {traza['epoch']}", expanded=traza["epoch"] == 0):
            st.caption(f"X shape {traza['X'].shape} · Y shape {traza['Y'].shape}")
            for i, capa_snap in enumerate(traza["capas"]):
                st.markdown(f"**{NOMBRES_CAPA[i]}**")
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"W {capa_snap['W'].shape}")
                    st.dataframe(df_matriz(capa_snap["W"], "w"), width="stretch")
                    st.write(f"z {capa_snap['z'].shape}")
                    st.dataframe(df_matriz(capa_snap["z"], "z"), width="stretch")
                with c2:
                    st.write(f"b {capa_snap['b'].shape}")
                    st.dataframe(df_matriz(capa_snap["b"], "b"), width="stretch")
                    st.write(f"a {capa_snap['a'].shape}")
                    st.dataframe(df_matriz(capa_snap["a"], "a"), width="stretch")
                st.write(f"delta {capa_snap['delta'].shape}")
                st.dataframe(df_matriz(capa_snap["delta"], "d"), width="stretch")
                st.divider()


# Sustituye circulo() del tutorial: puntos GPS alrededor de un centro.
def circulo(num_datos=100, lat_centro=0.0, lon_centro=0.0, radio=8.0):
    r = radio * np.sqrt(np.random.rand(num_datos))
    theta = np.random.rand(num_datos) * 2 * math.pi
    lat = lat_centro + r * np.sin(theta)
    lon = lon_centro + (r * np.cos(theta)) * 1.4
    lat = np.clip(lat, -90, 90)
    lon = np.clip(lon, -180, 180)
    return np.column_stack([np.round(lat, 3), np.round(lon, 3)])


MONTERREY = (25.6866, -100.3161)
LILLE = (50.6292, 3.0573)
DATASET_ID = "monterrey-lille-csv-v1"
DIR_DATOS = Path(__file__).resolve().parent
CSV_MONTERREY = DIR_DATOS / "monterrey.csv"
CSV_LILLE = DIR_DATOS / "lille.csv"


def cargar_csv_ciudad(ruta):
    df = pd.read_csv(ruta)
    return df[["lat", "lon"]].to_numpy(dtype=float)


def generar_datasets(n_por_clase=150, seed=42):
    monterrey = cargar_csv_ciudad(CSV_MONTERREY)
    lille = cargar_csv_ciudad(CSV_LILLE)
    X = np.vstack([monterrey, lille])
    Y = np.array([0] * len(monterrey) + [1] * len(lille)).reshape(-1, 1)
    return X, Y


def normalizar(X, media=None, std=None):
    if media is None:
        media = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
    return (X - media) / std, media, std


def dataframe_mapa(X, Y):
    df = pd.DataFrame(X, columns=["lat", "lon"])
    df["clase"] = np.where(Y.ravel() == 0, "Monterrey", "Lille")
    df["color"] = np.where(Y.ravel() == 0, "#1f77b4", "#d62728")
    return df


def mostrar_mapa(df, centro, zoom=12, height=440):
    """Mapa Leaflet + OpenStreetMap, sin token de Mapbox/Carto."""
    m = folium.Map(
        location=[centro[0], centro[1]],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    for row in df.itertuples():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=5,
            color=row.color,
            fill=True,
            fill_color=row.color,
            fill_opacity=0.75,
            weight=1,
            popup=f"{row.clase}: {row.lat}, {row.lon}",
        ).add_to(m)
    components.html(m.get_root().render(), height=height, scrolling=False)


# App Streamlit
st.set_page_config(page_title="Red neuronal de ciudades", layout="wide")
st.title("Clasificador de ciudades")
st.header("Monterrey vs Lille")
st.caption(
    "Red neuronal programada desde 0 en NumPy. "
    "150 puntos alrededor de Monterrey y 150 alrededor de Lille."
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
