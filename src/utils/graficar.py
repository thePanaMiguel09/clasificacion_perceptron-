import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
COLORES_PUNTOS  = ["#4C9BE8", "#FF6B6B", "#3DDC84"]
COLORES_PLANOS  = ["#1A6FBF", "#CC2E2E", "#1A9E52"]   
NOMBRES_CLASES  = ["Clase 0", "Clase 1", "Clase 2"]
ALPHA_PUNTOS    = 0.75
ALPHA_PLANO     = 0.18  
ALPHA_BORDE     = 0.55  
TAMANO_PUNTO    = 18


def _estilo_axes3d(ax):
    ax.set_facecolor("#0D1117")
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor("#161B22")
    ax.yaxis.pane.set_facecolor("#161B22")
    ax.zaxis.pane.set_facecolor("#161B22")
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine._axinfo["grid"]["color"] = "#30363D"
        spine._axinfo["grid"]["linewidth"] = 0.5
    ax.tick_params(colors="#8B949E", labelsize=7)
    ax.xaxis.label.set_color("#8B949E")
    ax.yaxis.label.set_color("#8B949E")
    ax.zaxis.label.set_color("#8B949E")


def _dibujar_plano(ax, w, b, x, color_fill, color_edge):
 
    if abs(w[2]) < 1e-6:
        return

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x3_min, x3_max = x[:, 2].min() - 1, x[:, 2].max() + 1

    x1_vals, x2_vals = np.meshgrid(
        np.linspace(x1_min, x1_max, 35),
        np.linspace(x2_min, x2_max, 35)
    )
    x3_vals = -(w[0] * x1_vals + w[1] * x2_vals + b) / w[2]
    x3_vals = np.clip(x3_vals, x3_min, x3_max)

    ax.plot_surface(
        x1_vals, x2_vals, x3_vals,
        alpha=ALPHA_PLANO,
        color=color_fill,
        edgecolor="none",
        antialiased=True,
        zorder=1,
    )

    ax.plot_wireframe(
        x1_vals, x2_vals, x3_vals,
        rstride=17, cstride=17,          
        color=color_edge,
        alpha=ALPHA_BORDE,
        linewidth=0.8,
        zorder=2,
    )

    for i in [0, -1]:
        ax.plot(x1_vals[i, :], x2_vals[i, :], x3_vals[i, :],
                color=color_edge, alpha=0.7, linewidth=1.0, zorder=3)
        ax.plot(x1_vals[:, i], x2_vals[:, i], x3_vals[:, i],
                color=color_edge, alpha=0.7, linewidth=1.0, zorder=3)

def graficar(X, y, historiales, n_clases, pesos=None):
    fig = plt.figure(figsize=(16, 6), facecolor="#0D1117")
    fig.subplots_adjust(left=0.04, right=0.96, wspace=0.32)

  
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_facecolor("#0D1117")
    _estilo_axes3d(ax1)

    for c in range(n_clases):
        mascara = y == c
        ax1.scatter(
            X[mascara, 0], X[mascara, 1], X[mascara, 2],
            c=COLORES_PUNTOS[c],
            label=NOMBRES_CLASES[c],
            alpha=ALPHA_PUNTOS,
            s=TAMANO_PUNTO,
            edgecolors="none",
            depthshade=True,
            zorder=4,
        )

    if pesos is not None:
        if n_clases == 2:
            w = np.array(pesos["clase_0"]["w"])
            b = pesos["clase_0"]["b"]
            _dibujar_plano(ax1, w, b, X,
                           color_fill=COLORES_PUNTOS[0],
                           color_edge=COLORES_PLANOS[0])
        else:
            for c in range(n_clases):
                w = np.array(pesos[f"clase_{c}"]["w"])
                b = pesos[f"clase_{c}"]["b"]
                _dibujar_plano(ax1, w, b, X,
                               color_fill=COLORES_PUNTOS[c],
                               color_edge=COLORES_PLANOS[c])

    ax1.set_xlabel("x₁", labelpad=4)
    ax1.set_ylabel("x₂", labelpad=4)
    ax1.set_zlabel("x₃", labelpad=4)
    ax1.set_title("Puntos 3D por clase", color="#E6EDF3",
                  fontsize=11, pad=10, fontweight="semibold")

   
    handles = [
        Patch(facecolor=COLORES_PUNTOS[c], label=NOMBRES_CLASES[c], alpha=0.9)
        for c in range(n_clases)
    ]
    ax1.legend(
        handles=handles,
        loc="upper left",
        framealpha=0.25,
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
        fontsize=8,
    )

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#0D1117")
    ax2.spines[:].set_color("#30363D")
    ax2.tick_params(colors="#8B949E", labelsize=8)
    ax2.xaxis.label.set_color("#8B949E")
    ax2.yaxis.label.set_color("#8B949E")
    ax2.grid(True, color="#21262D", linewidth=0.6, linestyle="--")

    for nombre_clase, historial in historiales.items():
        # compatible con clave "clase_0" y "Clase_0"
        raw = nombre_clase.split('_')[-1]
        idx = int(raw) if raw.isdigit() else 0
        epocas = range(1, len(historial) + 1)

        # Sombra debajo de la curva
        ax2.fill_between(epocas, historial,
                         alpha=0.12, color=COLORES_PUNTOS[idx])
        # Línea principal
        ax2.plot(epocas, historial,
                 color=COLORES_PUNTOS[idx],
                 label=nombre_clase,
                 linewidth=2.2,
                 solid_capstyle="round")

    ax2.set_xlabel("Época", fontsize=9)
    ax2.set_ylabel("Tasa de error", fontsize=9)
    ax2.set_title("Error vs Época", color="#E6EDF3",
                  fontsize=11, fontweight="semibold")
    ax2.set_ylim(0, 1)
    ax2.legend(
        framealpha=0.25,
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
        fontsize=8,
    )

    plt.savefig("verificacion.png", dpi=140,
                facecolor="#0D1117", bbox_inches="tight")
    plt.show()
    print("[OK] Gráfica guardada como verificacion.png")


def animar_plano(x, y, historial_pesos, n_clases, intervalo_ms=200):
    fig = plt.figure(figsize=(9, 7), facecolor="#0D1117")
    ax  = fig.add_subplot(111, projection="3d")
    _estilo_axes3d(ax)

    for c in range(n_clases):
        mascara = y == c
        ax.scatter(
            x[mascara, 0], x[mascara, 1], x[mascara, 2],
            c=COLORES_PUNTOS[c], label=NOMBRES_CLASES[c],
            alpha=0.55, s=12, edgecolors="none", depthshade=True
        )

    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_zlabel("x₃")
    handles = [Patch(facecolor=COLORES_PUNTOS[c], label=NOMBRES_CLASES[c])
               for c in range(n_clases)]
    ax.legend(handles=handles, loc="upper left",
              framealpha=0.25, facecolor="#161B22",
              edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=8)

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x3_min, x3_max = x[:, 2].min() - 1, x[:, 2].max() + 1

    x1_vals, x2_vals = np.meshgrid(
        np.linspace(x1_min, x1_max, 35),
        np.linspace(x2_min, x2_max, 35)
    )

    snapshots_por_clase = (
        {"clase_0": historial_pesos}
        if isinstance(historial_pesos, list)
        else historial_pesos
    )
    n_epocas   = len(list(snapshots_por_clase.values())[0])
    superficies = {k: [] for k in snapshots_por_clase}
    titulo      = ax.set_title("", color="#E6EDF3", fontsize=10)

    def actualizar(frame):
        for clave, surfs in superficies.items():
            for s in surfs:
                s.remove()
            surfs.clear()

        for clave, snapshots in snapshots_por_clase.items():
            raw = clave.split('_')[-1]
            idx = int(raw) if raw.isdigit() else 0
            snap = snapshots[frame]
            w, b = snap["w"], snap["b"]

            if abs(w[2]) > 1e-6:
                x3_vals = -(w[0] * x1_vals + w[1] * x2_vals + b) / w[2]
                x3_vals = np.clip(x3_vals, x3_min, x3_max)

                s1 = ax.plot_surface(x1_vals, x2_vals, x3_vals,
                                     alpha=ALPHA_PLANO,
                                     color=COLORES_PUNTOS[idx],
                                     edgecolor="none", antialiased=True)
                s2 = ax.plot_wireframe(x1_vals, x2_vals, x3_vals,
                                       rstride=17, cstride=17,
                                       color=COLORES_PLANOS[idx],
                                       alpha=ALPHA_BORDE, linewidth=0.8)
                superficies[clave].extend([s1, s2])

        titulo.set_text(f"Época {frame + 1} / {n_epocas}")
        return []

    anim = FuncAnimation(fig, actualizar, frames=n_epocas,
                         interval=intervalo_ms, repeat=True)

    print("[...] Guardando animación...")
    anim.save("entrenamiento.gif", writer="pillow", fps=5,
              savefig_kwargs={"facecolor": "#0D1117"})
    print("[OK] Animación guardada como entrenamiento.gif")
    plt.tight_layout()
    plt.show()


def graficar_pesos(historial_pesos, n_clases, historiales_error=None):
    COLORES_W = ["#4C9BE8", "#FF6B6B", "#3DDC84", "#FFB347"]
    NOMBRES_W = ["w₁", "w₂", "w₃", "b (bias)"]

    snapshots_por_clase = (
        {"clase_0": historial_pesos}
        if isinstance(historial_pesos, list)
        else historial_pesos
    )
    n_filas = len(snapshots_por_clase)

    fig, axes = plt.subplots(n_filas, 1,
                             figsize=(13, 4.5 * n_filas),
                             facecolor="#0D1117",
                             squeeze=False)

    for fila, (nombre_clase, snapshots) in enumerate(snapshots_por_clase.items()):
        ax = axes[fila][0]
        ax.set_facecolor("#0D1117")
        ax.spines[:].set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=8)
        ax.xaxis.label.set_color("#8B949E")
        ax.yaxis.label.set_color("#8B949E")
        ax.grid(True, color="#21262D", linewidth=0.6, linestyle="--")

        epocas = [s["epoca"] for s in snapshots]
        series = [
            [s["w"][0] for s in snapshots],
            [s["w"][1] for s in snapshots],
            [s["w"][2] for s in snapshots],
            [s["b"]    for s in snapshots],
        ]

        for vals, color, nombre in zip(series, COLORES_W, NOMBRES_W):
            ax.plot(epocas, vals, color=color, linewidth=2.0,
                    label=nombre, marker="o", markersize=2.5,
                    solid_capstyle="round",
                    linestyle="--" if nombre.startswith("b") else "-")

        ax.set_xlabel("Época", fontsize=9)
        ax.set_ylabel("Valor del parámetro", fontsize=9)
        ax.set_title(f"Evolución de pesos — {nombre_clase}",
                     color="#E6EDF3", fontsize=11, fontweight="semibold")
        ax.legend(framealpha=0.25, facecolor="#161B22",
                  edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=8)

        if historiales_error and nombre_clase in historiales_error:
            ax2 = ax.twinx()
            ax2.set_facecolor("#0D1117")
            ax2.spines[:].set_color("#30363D")
            ax2.tick_params(colors="#8B949E", labelsize=8)
            ax2.yaxis.label.set_color("#8B949E")
            errores = historiales_error[nombre_clase]
            ax2.plot(range(1, len(errores) + 1), errores,
                     color="#8B949E", linewidth=1.4,
                     label="Error", linestyle=":", alpha=0.8)
            ax2.fill_between(range(1, len(errores) + 1), errores,
                             alpha=0.06, color="#8B949E")
            ax2.set_ylabel("Tasa de error", fontsize=9)
            ax2.set_ylim(0, max(errores) * 1.5 if max(errores) > 0 else 1)
            ax2.legend(loc="upper left", framealpha=0.25,
                       facecolor="#161B22", edgecolor="#30363D",
                       labelcolor="#E6EDF3", fontsize=8)

    plt.tight_layout()
    plt.savefig("pesos_por_epoca.png", dpi=140,
                facecolor="#0D1117", bbox_inches="tight")
    plt.show()
    print("[OK] Gráfica de pesos guardada como pesos_por_epoca.png")
    