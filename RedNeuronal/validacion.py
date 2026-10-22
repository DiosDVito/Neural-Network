#Daniel Esparza Arizpe - A01637076
#Validación de las épocas 0 y 1: trazas y archivo de salida.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from modelo import crear_red, mse

NOMBRES_CAPA = ["Capa 1 (2→4, ReLU)", "Capa 2 (4→8, ReLU)", "Capa 3 (8→1, Sigmoide)"]
RUTA_VALIDACION = Path(__file__).resolve().parent / "validacion_epocas_0_1.txt"


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
