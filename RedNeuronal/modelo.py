#Daniel Esparza Arizpe - A01637076
#Red neuronal desde 0: activaciones, capas, forward, backprop y MSE.

from __future__ import annotations

import numpy as np
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
