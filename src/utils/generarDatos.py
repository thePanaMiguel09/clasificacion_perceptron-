import numpy as np

# Distribuciones disponibles
DISTRIBUCIONES = ["gaussiana", "uniforme", "exponencial", "laplace"]


def generarDatos(n_por_clase=200, sigma=1.0, n_clases=2,
                 semilla=None, distribucion="gaussiana"):
    """
    Genera nubes de puntos en R³ usando la distribución indicada.

    Parámetros
    ----------
    n_por_clase   : puntos por clase
    sigma         : parámetro de dispersión (escala) para todas las distribuciones
    n_clases      : 2 o 3
    semilla       : semilla aleatoria para reproducibilidad
    distribucion  : "gaussiana" | "uniforme" | "exponencial" | "laplace"
    """
    if semilla is not None:
        np.random.seed(semilla)

    distribucion = distribucion.lower().strip()
    if distribucion not in DISTRIBUCIONES:
        raise ValueError(f"Distribución '{distribucion}' no válida. "
                         f"Opciones: {DISTRIBUCIONES}")

    # ── Centros por clase ─────────────────────────────────────────
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

    x_lista = []
    y_lista = []

    for i, centro in enumerate(centros):
        puntos = _muestrear(distribucion, centro, sigma, n_por_clase)
        x_lista.append(puntos)
        y_lista.append(np.full(n_por_clase, i, dtype=int))

    x = np.vstack(x_lista)
    y = np.concatenate(y_lista)

    # Mezclar para evitar aprendizaje sesgado
    idxs = np.random.permutation(len(y))
    return x[idxs], y[idxs]


# ── Muestreadores por distribución ────────────────────────────────

def _muestrear(distribucion, centro, sigma, n):
    """Genera n puntos 3D centrados en `centro` con escala `sigma`."""
    if distribucion == "gaussiana":
        # N(centro, sigma²) — campana de Gauss clásica
        return np.random.normal(loc=centro, scale=sigma, size=(n, 3))

    elif distribucion == "uniforme":
        # U(centro - a, centro + a)  donde a = sigma * √3
        # Este factor hace que la varianza sea sigma² (igual que la gaussiana)
        a = sigma * np.sqrt(3)
        return centro + np.random.uniform(low=-a, high=a, size=(n, 3))

    elif distribucion == "exponencial":
        # Exp(λ) centrada: se resta la media (1/λ = sigma) para centrar en `centro`
        # Los puntos se distribuyen asimétricamente alrededor del centro
        muestras = np.random.exponential(scale=sigma, size=(n, 3))
        return centro + muestras - sigma   # restar media para centrar

    elif distribucion == "laplace":
        # Laplace(centro, b) donde b = sigma/√2
        # Colas más pesadas que la gaussiana — más puntos alejados del centro
        b = sigma / np.sqrt(2)
        return np.random.laplace(loc=centro, scale=b, size=(n, 3))