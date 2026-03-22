import json



def exportarJSON(X, y, historiales, tiempo, sigma, n_clases, archivo= "datos.json"):
    payload ={
        "metadata": {
            "sigma": round(float(sigma), 4),
            "n_clases": int(n_clases),
            "n_puntos_total": int(len(y)),
            "tiempo_segundos": round(float(tiempo), 6),
        },
        "error_por_epoca": historiales,
        "puntos": [
            {
                "x": round(float(X[i, 0]), 4),
                "y": round(float(X[i, 1]), 4),
                "z": round(float(X[i, 2]), 4),
                "clase": int(y[i])
            }
            for i in range(len(y))
        ]
    }

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] JSON exportado -> {archivo}")
        print(f"    Puntos: {len(y)} | Tiempo: {tiempo: .4f}s | sigma={sigma}")