import numpy as np


def generarDatos(n_por_clase= 200, sigma= 1.0, n_clases=2, semilla= None):
    if semilla is not None:
        np.random.seed(semilla)

    if n_clases == 2:
        centros = [
            np.array([1.0, 1.0, 1.0]),
            np.array([6.0, 6.0, 6.0])
        ]
    else:
        centros = [
            np.array([1.0, 1.0, 1.0]),
            np.array([8.0, 1.0, 1.0]),
            np.array([4.5, 8.0, 8.0])
        ]

    x_lista =[]
    y_lista=[]

    for i, centro in enumerate(centros):
        puntos = np.random.normal(loc=centro, scale=sigma, size=(n_por_clase, 3))
        etiquetas = np.full(n_por_clase, i, dtype=int)

        x_lista.append(puntos)
        y_lista.append(etiquetas)

    x = np.vstack(x_lista)
    y = np.concatenate(y_lista)

    idxs = np.random.permutation(len(y))
    return x[idxs], y[idxs]