"""
run_demo.py

Este script es el punto de entrada para ejecutar una demostración del Algoritmo Genético Elitista (EGA).

Carga la configuración desde un archivo YAML, inicializa el evaluador "de juguete" y el
algoritmo genético, y luego ejecuta el proceso de optimización para encontrar los
mejores parámetros para el modelo de juguete.

Uso:
    python run_demo.py --config config.yaml
"""
import yaml, argparse, os
from ega_core import EGA
from evaluator_toy import ToyODEEvaluator
import numpy as np

def load_config(path):
    """Carga un archivo de configuración en formato YAML."""
    with open(path, "r") as filehandler:
        config = yaml.safe_load(filehandler)
    return config
"""
def get_default_config():
    # Retorna la configuración por defecto para la demostración.
    default_bounds = [[0.1, 3.0], [0.01, 1.0], [-3.0, 3.0]] * 3
    return {
        # Parámetros del evaluador
        "target": [1.0, 0.8, 0.6],
        "t_span": (0, 50),
        "dt": 0.5,
        "noise_std": 0.0,
        "bounds": default_bounds,
        # Parámetros del EGA
        "populationSize": 40,
        "generations": 60,
        "crossover_rate": 0.7,
        "mutation_rate": 0.15,
        "elite_size": 2,
        "alpha_blx": 0.3,
        "mutation_scale": [0.05] * len(default_bounds),
        "timeout": 15.0,
        "processes": max(1, os.cpu_count() - 1),
        "seed": 42,
        "tournament_k": 3,
        # Otros parámetros
        "snapshot_dir": "snapshots"
    }
"""
def main():
    """Función principal que ejecuta la demostración."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()
    
    # Cargar configuración desde el archivo y fusionarla con la configuración por defecto
    user_config = load_config(args.config)
    # config = {**get_default_config(), **user_config}
    config = { **user_config }

    # Instanciar el evaluador con los parámetros de la configuración
    evaluator = ToyODEEvaluator(
        target=np.array(config["target"], dtype=float),
        t_span=tuple(config["t_span"]),
        dt=config["dt"],
        noise_std=config["noise_std"]
    )

    # Configuración del EGA
    ega_config = {
        key: config[key] for key in [
            "populationSize", "generations", "crossover_rate", "mutation_rate", 
            "elite_size", "bounds", "alpha_blx", "mutation_scale", "timeout", 
            "processes", "seed", "tournament_k"
        ]
    }

    print("Configuración del EGA:", ega_config)
    
    # Crear y ejecutar el algoritmo genético
    ega = EGA(ega_config, evaluator)
    res = ega.run(snapshot_dir=config["snapshot_dir"], verbose=True)
    
    # Imprimir resultados
    print("Mejor solución encontrada:", res["best"])
    print("Tiempo total (s):", res["total_time_s"])

if __name__ == "__main__":
    # Punto de entrada del script.
    main()
