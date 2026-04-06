import json


def exportarJSON(
        X,
        y,
        historiales,
        tiempo,
        sigma,
        n_clases,
        semilla=None,
        pesos=None,
        historial_pesos=None,
        distribucion="gaussiana",
        archivo="datos.json"):

    historial_pesos_serial = {}
    if historial_pesos is not None:
        for nombre_clase, snapshots in historial_pesos.items():
            historial_pesos_serial[nombre_clase] = [
                {
                    "epoca": snap["epoca"],
                    "w":     [round(float(v), 6) for v in snap["w"]],
                    "b":     round(float(snap["b"]), 6)
                }
                for snap in snapshots
            ]

    payload = {
        "metadata": {
            "sigma":           round(float(sigma), 4),
            "n_clases":        int(n_clases),
            "n_puntos_total":  int(len(y)),
            "tiempo_segundos": round(float(tiempo), 6),
            "semilla":         semilla,
            "distribucion":    distribucion       # ← NUEVO
        },
        "pesos":            pesos,
        "historial_pesos":  historial_pesos_serial,
        "error_por_epoca":  historiales,
        "puntos": [
            {
                "x":     round(float(X[i, 0]), 4),
                "y":     round(float(X[i, 1]), 4),
                "z":     round(float(X[i, 2]), 4),
                "clase": int(y[i])
            }
            for i in range(len(y))
        ],
    }

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] JSON exportado -> {archivo}")
    print(f"    Puntos: {len(y)} | Tiempo: {tiempo:.4f}s | "
          f"sigma={sigma} | dist={distribucion}")