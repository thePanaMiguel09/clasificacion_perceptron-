import matplotlib.pyplot as plt


def graficar(X, y, historiales, n_clases):
    colores = ["royalblue", "tomato", "mediumseagreen"]
    nombres = ["Clase 0", "Clase 1", "Clase 2"]

    fig = plt.figure(figsize=(14, 5))

    aux1 = fig.add_subplot(121, projection="3d")

    for c in range(n_clases):
        mascara = y == c
        aux1.scatter(
            X[mascara, 0],
            X[mascara, 1],
            X[mascara, 2],
            c = colores[c], label=nombres[c],
            alpha=0.6, s=15
        )

    
    aux1.set_xlabel("x1")
    aux1.set_ylabel("x2")
    aux1.set_zlabel("x3")
    aux1.set_title("Puntos 3D por clase")
    aux1.legend()

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