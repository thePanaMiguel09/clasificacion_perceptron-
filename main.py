import os
import json
import argparse
from src.utils import generarDatos, exportarJSON, graficar, animar_plano, graficar_pesos
from src.perceptron import Perceptron
from src.perceptron_one_vs_rest import PerceptronOneVsRest


def main():
    parser = argparse.ArgumentParser(description="Perceptrón con visualización Unity")

    parser.add_argument("--sigma",        type=float, default=1.0)
    parser.add_argument("--clases",       type=int,   default=2, choices=[2, 3])
    parser.add_argument("--epocas",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--n",            type=int,   default=200)
    parser.add_argument("--semilla",      type=int,   default=None)
    parser.add_argument("--distribucion", type=str,   default="gaussiana",
                        choices=["gaussiana", "uniforme", "exponencial", "laplace"])
    parser.add_argument("--graficar",  action="store_true")
    parser.add_argument("--animar",    action="store_true")
    parser.add_argument("--pesos",     action="store_true")
    parser.add_argument("--salida",    type=str, default="datos.json")

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Perceptrón — {args.clases} clases | dist={args.distribucion}")
    print(f"  Sigma={args.sigma}  |  lr={args.lr}  |  epocas={args.epocas}")
    print(f"{'='*50}\n")

    print("[1/3] Generando datos...")
    X, y = generarDatos(
        n_por_clase=args.n,
        sigma=args.sigma,
        n_clases=args.clases,
        semilla=args.semilla,
        distribucion=args.distribucion
    )
    print(f"  {len(y)} puntos | dist={args.distribucion} | sigma={args.sigma}")

    carpeta_salida   = os.path.dirname(os.path.abspath(args.salida))
    archivo_progreso = os.path.join(carpeta_salida, "progreso.json")

    with open(archivo_progreso, "w") as f:
        json.dump({"entrenando": False, "epoca_actual": 0,
                   "epocas_total": args.epocas, "clases": {}}, f)

    print("[2/3] Entrenando modelo...")

    if args.clases == 2:
        model = Perceptron(n_entradas=3, lr=args.lr, epocas=args.epocas)
        historial, historial_pesos, tiempo = model.entrenar_perceptron(
            x=X, y=y,
            archivo_progreso=archivo_progreso,
            nombre_clase="clase_0"
        )
        historiales          = {"clase_0": historial}
        historial_pesos_dict = {"clase_0": historial_pesos}
        pesos_finales        = {"clase_0": {"w": model.w.tolist(), "b": float(model.b)}}
    else:
        model = PerceptronOneVsRest(
            n_clases=3, n_entradas=3, lr=args.lr, epocas=args.epocas
        )
        historiales, historial_pesos, tiempo = model.entrenar(
            x=X, y=y, archivo_progreso=archivo_progreso
        )
        historial_pesos_dict = historial_pesos
        pesos_finales = {
            f"clase_{c}": {"w": p.w.tolist(), "b": float(p.b)}
            for c, p in enumerate(model.perceptrones)
        }

    with open(archivo_progreso, "r", encoding="utf-8") as f:
        ultimo = json.load(f)
    ultimo["entrenando"] = False
    with open(archivo_progreso, "w", encoding="utf-8") as f:
        json.dump(ultimo, f)

    error_final = list(historiales.values())[0][-1]
    print(f"  Tiempo: {tiempo:.4f}s | Error final: {error_final:.1%}")

    print("[3/3] Exportando resultados...")
    exportarJSON(
        X=X,
        y=y,
        historiales=historiales,
        tiempo=tiempo,
        sigma=args.sigma,
        n_clases=args.clases,
        semilla=args.semilla,
        pesos=pesos_finales,
        historial_pesos=historial_pesos_dict,
        distribucion=args.distribucion,
        archivo=args.salida,
    )

    if args.graficar:
        graficar(X, y, historiales, args.clases, pesos=pesos_finales)
    if args.animar:
        animar_plano(X, y, historial_pesos_dict, n_clases=args.clases)
    if args.pesos:
        graficar_pesos(historial_pesos_dict, n_clases=args.clases,
                       historiales_error=historiales)

    print("[OK] Entrenamiento finalizado.")


if __name__ == "__main__":
    main()