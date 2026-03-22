from src.utils import generarDatos, exportarJSON, graficar
from src.perceptron import Perceptron
from src.perceptron_one_vs_rest import PerceptronOneVsRest
import argparse

def main():
    parser = argparse.ArgumentParser(description="Perceptrón con visualización Unity")

    parser.add_argument('--sigma', type=float, default=1.0, help="Desviación estándar de datos")

    parser.add_argument("--clases", type=int, default=2, choices=[2, 3])

    parser.add_argument("--epocas", type=int, default=50, help="Número de épocas de entrenamiento (detault: 50)")

    parser.add_argument("--lr", type=float, default= 0.1, help="Tasa de aprendizaje (default: 0.1)")

    parser.add_argument("--n", type=int, default= 200, help="Número de puntos por clase (Default: 200)" )

    parser.add_argument("--semilla", type=int, default=None, help="Semilla aleatoria para reproducibilidad")

    parser.add_argument("--graficar", action='store_true', help="Mostrar graficas en Python antes de exportar")

    parser.add_argument("--salida", type=str, default="datos.json", help="Nombre del archivo JSON de salida")

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Perceptrón — Escenario {args.clases} clases")
    print(f"  σ={args.sigma}  |  lr={args.lr}  |  épocas={args.epocas}")
    print(f"{'='*50}\n")

    print("[1/3] Generando datos..")


    X, y= generarDatos(
            n_por_clase= args.n,
            sigma= args.sigma,
            n_clases= args.clases,
            semilla=args.semilla
    )

    print(f" {len(y)} puntos generados ({args.clases} clases × {args.n} puntos)")


    print("[2/3] Entrenando modelo")

    if args.clases == 2:
        model = Perceptron(epocas=args.epocas, n_entradas=3, lr= args.lr)
        historial, tiempo = model.entrenar_perceptron(x=X, y=y)
        historiales = {"clase_0": historial}
    else:
        model = PerceptronOneVsRest(n_clases=3, n_entradas=3, lr=args.lr, epocas=args.epocas)
        historiales, tiempo = model.entrenar(x=X, y=y)

    error_final = list(historiales.values())[0][-1]
    print(f"Tiempo de entrenamiento: {tiempo:.4f}s | Error final: {error_final:.1%}")

    print("[3/3] Exportando resultados..")

    exportarJSON(X=X, y=y, historiales= historiales, n_clases=args.clases, tiempo= tiempo, semilla = args.semilla, sigma=args.sigma, archivo=args.salida)

    if args.graficar:
        graficar(X, y, historiales, args.clases)

    print("[OK] Entrenamiento finalizado.")

if __name__ == "__main__":
    main()








    

