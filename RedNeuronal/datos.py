#Daniel Esparza Arizpe - A01637076
#Puntos GPS: lectura de CSV por ciudad y normalización.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

MONTERREY = (25.6866, -100.3161)
LILLE = (50.6292, 3.0573)
DATASET_ID = "monterrey-lille-csv-v1"
DIR_DATOS = Path(__file__).resolve().parent
CSV_MONTERREY = DIR_DATOS / "monterrey.csv"
CSV_LILLE = DIR_DATOS / "lille.csv"


def circulo(num_datos=100, lat_centro=0.0, lon_centro=0.0, radio=8.0):
    """Genera puntos GPS alrededor de un centro (versión adaptada del PDF)."""
    r = radio * np.sqrt(np.random.rand(num_datos))
    theta = np.random.rand(num_datos) * 2 * math.pi
    lat = lat_centro + r * np.sin(theta)
    lon = lon_centro + (r * np.cos(theta)) * 1.4
    lat = np.clip(lat, -90, 90)
    lon = np.clip(lon, -180, 180)
    return np.column_stack([np.round(lat, 3), np.round(lon, 3)])


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
