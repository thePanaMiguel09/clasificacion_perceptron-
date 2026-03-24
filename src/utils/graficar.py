import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def _dibujar_plano(ax, w, b, x):
    if abs(w[2]) < 1e-6:
        return

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x3_min, x3_max = x[:, 2].min() - 1, x[:, 2].max() + 1

    x1_vals, x2_vals = np.meshgrid(
        np.linspace(x1_min, x1_max, 20),
        np.linspace(x2_min, x2_max, 20)
    )

    x3_vals = -(w[0] * x1_vals + w[1] * x2_vals + b) / w[2]

    x3_vals = np.clip(x3_vals, x3_min, x3_max)

    ax.plot_surface(
        x1_vals, x2_vals, x3_vals,
        alpha=0.25,
        color="gray",
        edgecolor="none"
    )  

def graficar(X, y, historiales, n_clases, pesos = None):
    colores = ["royalblue", "tomato", "mediumseagreen"]
    nombres = ["Clase 0", "Clase 1", "Clase 2"]

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection="3d")

    for c in range(n_clases):
        mascara = y == c
        ax1.scatter(
            X[mascara, 0],
            X[mascara, 1],
            X[mascara, 2],
            c = colores[c], label=nombres[c],
            alpha=0.6, s=15
        )

    if pesos is not None:
        if n_clases == 2:
            # Un solo plano que separa clase 0 de clase 1
            w = np.array(pesos["clase_0"]["w"])
            b = pesos["clase_0"]["b"]
            _dibujar_plano(ax1, w, b, X)
        else:
            # Para 3 clases dibujamos los 3 planos, uno por color
            for c in range(n_clases):
                w = np.array(pesos[f"clase_{c}"]["w"])
                b = pesos[f"clase_{c}"]["b"]
                _dibujar_plano(ax1, w, b, X)

    
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("x3")
    ax1.set_title("Puntos 3D por clase")
    ax1.legend()

    aux2 = fig.add_subplot(122)
    for nombre_clase, historial in historiales.items():
        idx = int(nombre_clase.split('_')[1])
        aux2.plot(
            range(1, len(historial) + 1),
            historial,
            color = colores[idx],
            label=nombre_clase,
            linewidth=2
        )

    aux2.set_xlabel("Época")
    aux2.set_ylabel("Tasa de error")
    aux2.set_title("Error vs Época")
    aux2.set_ylim(0, 1)
    aux2.legend()
    aux2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig("verificacion.png", dpi=120)
    plt.show()
    print("[OK] Gráfica guardada como verificacion.png")

def animar_plano(x, y, historial_pesos, n_clases, intervalo_ms=200):
    colores = ["royalblue", "tomato", "mediumseagreen"]
    nombres = ["Clase 0", "Clase 1", "Clase 2"]

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    for c in range(n_clases):
        mascara = y == c
        ax.scatter(
            x[mascara, 0], x[mascara, 1], x[mascara, 2],
            c= colores[c], label=nombres[c], alpha=0.5, s=12 
        )
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend(loc="upper left")

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x3_min, x3_max = x[:, 2].min() - 1, x[:, 2].max() + 1

    x1_vals, x2_vals = np.meshgrid(
        np.linspace(x1_min, x1_max, 20),
        np.linspace(x2_min, x2_max, 20)
    )

    if isinstance(historial_pesos, list):
        snapshots_por_clase = {"clase_0": historial_pesos}
    else:
        snapshots_por_clase = historial_pesos

    n_epocas = len(list(snapshots_por_clase.values())[0])

    superficies = {clave: None for clave in snapshots_por_clase}

    titulo = ax.set_title("")

    def actualizar(frame):
        for clave in superficies:
            if superficies[clave] is not None:
                superficies[clave].remove()
                superficies[clave] = None

        for clave, snapshots in snapshots_por_clase.items():
            idx = int(clave.split('_')[1])
            snap = snapshots[frame]
            w    = snap["w"]
            b    = snap["b"]

            if abs(w[2]) > 1e-6:
                x3_vals = -(w[0] * x1_vals + w[1] * x2_vals + b) / w[2]
                x3_vals = np.clip(x3_vals, x3_min, x3_max)

                superficies[clave] = ax.plot_surface(
                    x1_vals, x2_vals, x3_vals,
                    alpha=0.25,
                    color=colores[idx],
                    edgecolor="none"
                )

        titulo.set_text(f"Época {frame + 1} / {n_epocas}")
        return []

    anim = FuncAnimation(
        fig,
        actualizar,
        frames=n_epocas,
        interval=intervalo_ms,
        repeat=True
    )

    print("[...] Guardando animación, esto puede tomar unos segundos...")
    anim.save("entrenamiento.gif", writer="pillow", fps=5)
    print("[OK] Animación guardada como entrenamiento.gif")

    plt.tight_layout()
    plt.show()

def graficar_pesos(historial_pesos, n_clases, historiales_error=None):

    colores_pesos = ["royalblue",  "tomato", "mediumseagreen", "darkorange"]
    nombres_pesos = ["w1", "w2", "w3", "b"]
    colores_clases = ["royal_blue", "tomato", "mediumseagreen"]

    if isinstance(historial_pesos, list):
        snapshots_por_clase = {"clase_0":historial_pesos}
    else:
        snapshots_por_clase = historial_pesos

    n_clases_reales = len(snapshots_por_clase)

    fig, axes = plt.subplots(
        n_clases_reales, 
        1,
        figsize=(12, 4 * n_clases_reales),
        squeeze=False
    )

    for fila, (nombre_clase, snapshots) in enumerate(snapshots_por_clase.items()):
        ax = axes[fila][0]

        epocas = [s["epoca"] for s in snapshots]

        w1 = [s["w"][0] for s in snapshots]
        w2 = [s["w"][1] for s in snapshots]
        w3 = [s["w"][2] for s in snapshots]
        b  = [s["b"]    for s in snapshots]

        ax.plot(epocas, w1, color=colores_pesos[0], linewidth=2,
                label="w₁", marker="o", markersize=3)
        ax.plot(epocas, w2, color=colores_pesos[1], linewidth=2,
                label="w₂", marker="o", markersize=3)
        ax.plot(epocas, w3, color=colores_pesos[2], linewidth=2,
                label="w₃", marker="o", markersize=3)
        ax.plot(epocas, b,  color=colores_pesos[3], linewidth=2,
                label="b (bias)", marker="o", markersize=3,
                linestyle="--")

        ax.set_xlabel("Época")
        ax.set_ylabel("Valor del parámetro")
        ax.set_title(f"Evolución de pesos — {nombre_clase}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)


        if historiales_error is not None and nombre_clase in historiales_error:
            ax2 = ax.twinx()
            errores = historiales_error[nombre_clase]
            ax2.plot(
                range(1, len(errores) + 1),
                errores,
                color="gray", linewidth=1.5,
                label="Error", linestyle=":",
                alpha=0.7
            )
            ax2.set_ylabel("Tasa de error", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")
            ax2.set_ylim(0, max(errores) * 1.5 if max(errores) > 0 else 1)
            ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("pesos_por_epoca.png", dpi=120)
    plt.show()
    print("[OK] Gráfica de pesos guardada como pesos_por_epoca.png")
    