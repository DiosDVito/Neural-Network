# Clasificador Monterrey vs Lille

Red neuronal programada **desde 0 en NumPy** (sin Keras ni TensorFlow) que clasifica puntos GPS de dos ciudades:

- clase `0` → Monterrey, México
- clase `1` → Lille, Francia

La interfaz es una app de **Streamlit** con mapas, entrenamiento, predicción y una sección de validación de las épocas 0 y 1.

Autor: Daniel Esparza Arizpe (A01637076)

## Cómo correrla

Desde la raíz del repositorio:

```bash
cd /Users/danielesparzarizpe/Developer/TEC/Compiladores
python3 -m venv .venv
.venv/bin/pip install -r RedNeuronal/requirements.txt
.venv/bin/streamlit run RedNeuronal/redneuronal.py
```

Se abre en [http://localhost:8501](http://localhost:8501). En la barra lateral pulsa **Entrenar red**. Más abajo está **Ejecutar validación**.

## Qué hace

1. Lee 150 puntos de `monterrey.csv` y 150 de `lille.csv`.
2. Normaliza latitud y longitud.
3. Entrena una red `[2, 4, 8, 1]` con ReLU, ReLU y sigmoide.
4. Muestra el MSE, el accuracy y una gráfica del error.
5. Permite predecir una coordenada nueva.
6. En validación, toma 20 puntos, corre solo las épocas 0 y 1 y guarda `X`, `W`, `b`, `z`, `a` y `delta` por capa en `validacion_epocas_0_1.txt`.

Arquitectura:

```text
lat, lon  →  ReLU[4]  →  ReLU[8]  →  Sigmoid[1]  →  0 = Monterrey, 1 = Lille
```

El accuracy suele llegar a ~100% porque las dos ciudades están muy lejos y casi no se solapan.

## Archivos

| Archivo | Rol |
|---|---|
| `redneuronal.py` | App Streamlit |
| `modelo.py` | Capa, activaciones, MSE, forward, backprop |
| `datos.py` | Lectura de CSV y normalización |
| `mapas.py` | Mapas Folium / OpenStreetMap |
| `validacion.py` | Trazas de épocas 0 y 1 |
| `monterrey.csv` | 150 puntos GPS de Monterrey (`lat`, `lon`) |
| `lille.csv` | 150 puntos GPS de Lille (`lat`, `lon`) |
| `validacion_epocas_0_1.txt` | Salida de la validación |
| `requirements.txt` | Dependencias |

Los puntos se toman de los CSV, no se generan al vuelo. Si editas esas coordenadas, recarga la app.

## Referencias

- [Cómo programar una red neuronal desde 0 en Python — Ander Fernández](https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/)
- [neural_network_numpy.py](https://github.com/adsoftsito/tecnologias-construccion/blob/main/neural_network_numpy.py)

La red (clase `capa`, ReLU/sigmoide, MSE, backpropagation y `circulo`) sigue esos códigos. Lo adaptado para el reto es usar GPS de dos ciudades, Streamlit y validar las primeras épocas.
