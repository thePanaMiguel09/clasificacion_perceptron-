import numpy as np
from src.perceptron import Perceptron


class PerceptronOneVsRest:
    def __init__(self, n_clases=3, n_entradas=3, lr=0.1, epocas=50):
        self.n_clases = n_clases
        self.perceptrones = [
            Perceptron(n_entradas=n_entradas, lr=lr, epocas=epocas)
            for _ in range(n_clases)
        ]

    def entrenar(self, x, y, archivo_progreso=None):
        historiales = {}
        historial_pesos = {}
        tiempo_total = 0.0

        for c, p in enumerate(self.perceptrones):
            y_binario = (y == c).astype(int)

            historial, snapshots, t = p.entrenar_perceptron(
                x,
                y_binario,
                archivo_progreso=archivo_progreso,
                nombre_clase=f"Clase_{c}",
            )
            historiales[f"Clase_{c}"] = historial
            historial_pesos[f"clase_{c}"] = snapshots
            tiempo_total += t

        return historiales, historial_pesos, tiempo_total

    def predecir_uno(self, x):
        scores = [p.predecir_score(x) for p in self.perceptrones]
        return int(np.argmax(scores))
