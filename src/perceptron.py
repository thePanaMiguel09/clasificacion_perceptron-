import numpy as np
import time

class Perceptron:
    def __init__(self, n_entradas=3, lr=0.1, epocas=50):
        self.w = np.random.randn(n_entradas) * 0.01
        self.b = 0.0
        self.lr = lr
        self.epocas = epocas

    def _activacion(self, z):
        return 1 if z >= 0 else 0

    def predecir_uno(self, x):
        z = np.dot(self.w, x) + self.b
        return self._activacion(z)

    def predecir_score(self, x):
        return np.dot(self.w, x) + self.b

    def entrenar_perceptron(self, x, y):
        historial_error = []
        inicio = time.perf_counter()

        for epoca in range(self.epocas):
            n_errores = 0

            for xi, yi in zip(x, y):
                y_pred = self.predecir_uno(xi)

                delta = int(yi) - y_pred

                if delta != 0:
                    self.w += self.lr * delta * xi
                    self.b += self.lr * delta
                    n_errores += 1
        
            tasa_error = n_errores/ len(y)
            historial_error.append(round(tasa_error, 6))
    
        tiempo_total = time.perf_counter() - inicio
        return historial_error, tiempo_total
            


    
