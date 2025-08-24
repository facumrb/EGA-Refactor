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
import warnings
from ega_core import EGA
from evaluator_toy import ToyODEEvaluator
import numpy as np

def load_config(path):
    """Carga un archivo de configuración en formato YAML."""
    with open(path, "r") as filehandler:
        config = yaml.safe_load(filehandler)
    return config

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
        "base_timeout": 15.0,
        "max_timeout": 60.0,
        "failure_rate_threshold_increase": 0.3,
        "failure_rate_threshold_decrease": 0.1,
        "timeout_increase_factor": 2.0,
        "timeout_decrease_factor": 0.8,
        "processes": max(1, os.cpu_count() - 1),
        "seed": 42,
        "tournament_k": 3,
        "fitness_penalty_factor": 0.001,
        "high_fitness_penalty": 1e6,
        "initial_conditions": [0.1, 0.1, 0.1],
        "min_production_rate": 1e-6, 
        "min_degradation_rate": 1e-3,
        # Otros parámetros
        "snapshot_dir": "snapshots"
    }

def validate_config(user_config, default_config):
    """Valida las claves en la configuración del usuario."""
    expected_keys = set(default_config.keys())
    user_keys = set(user_config.keys())
    
    # Chequeo de claves requeridas
    required_keys = expected_keys  # Asumimos todas son requeridas; ajusta si es necesario
    missing_keys = required_keys - user_keys
    if missing_keys:
        raise ValueError(f"Claves requeridas faltantes en config.yaml: {missing_keys}")
    
    # Alerta de claves desconocidas
    unknown_keys = user_keys - expected_keys
    if unknown_keys:
        warnings.warn(f"Claves desconocidas en config.yaml: {unknown_keys}. Serán ignoradas.")

def main():
    """Función principal que ejecuta la demostración."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()
    
    # Cargar configuración desde el archivo y fusionarla con la configuración por defecto
    user_config = load_config(args.config)
    default_config = get_default_config()
    validate_config(user_config, default_config)
    config = {**default_config, **user_config}

    # Parámetros del evaluador
    evaluator_config = {
        key: config[key] for key in [
            "target", "bounds", "t_span", "dt", "noise_std", 
            "fitness_penalty_factor", "high_fitness_penalty", "initial_conditions",
            "min_production_rate", "min_degradation_rate", "seed", "timeout"
        ]
    }

    # Instanciar el evaluador con los parámetros de la configuración
    evaluator = ToyODEEvaluator(evaluator_config)

    # Configuración del EGA
    ega_config = {
        key: config[key] for key in [
            "populationSize", "generations", "crossover_rate", "mutation_rate", 
            "elite_size", "bounds", "alpha_blx", "mutation_scale", "timeout", 
            "base_timeout", "max_timeout", "failure_rate_threshold_increase", "failure_rate_threshold_decrease",
            "timeout_increase_factor", "timeout_decrease_factor", "processes", "seed", "tournament_k", 
            "high_fitness_penalty"
        ]
    }

    print("Configuración del EGA:", ega_config)
    
    # Crear y ejecutar el algoritmo genético
    ega = EGA(ega_config, evaluator)
    results = ega.run(snapshot_dir=config["snapshot_dir"], verbose=True)
    
    # Imprimir resultados
    print("Mejor solución encontrada:", results["best"])
    print("Tiempo total (s):", results["total_time_s"])
    
if __name__ == "__main__":
    # Punto de entrada del script.
    main()
