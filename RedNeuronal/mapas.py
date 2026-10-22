#Daniel Esparza Arizpe - A01637076
#Mapas Leaflet de los puntos GPS.

from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import streamlit.components.v1 as components


def dataframe_mapa(X, Y):
    df = pd.DataFrame(X, columns=["lat", "lon"])
    df["clase"] = np.where(Y.ravel() == 0, "Monterrey", "Lille")
    df["color"] = np.where(Y.ravel() == 0, "#1f77b4", "#d62728")
    return df


def mostrar_mapa(df, centro, zoom=12, height=440):
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
