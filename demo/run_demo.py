"""
run_demo.py

Este script es el punto de entrada para ejecutar una demostración del Algoritmo Genético Elitista (EGA).

Carga la configuración desde un archivo YAML, inicializa el evaluador "de juguete" y el
algoritmo genético, y luego ejecuta el proceso de optimización para encontrar los
mejores parámetros para el modelo de juguete.

Uso:
    python run_demo.py --config config.yaml
"""
import yaml, argparse, os, sys
import warnings
from ega_core import EGA
from evaluator_toy import ToyODEEvaluator
import numpy as np
import pprint

def load_config(path):
    """Carga un archivo de configuración en formato YAML."""
    with open(path, "r") as filehandler:
        config = yaml.safe_load(filehandler)
    return config if config is not None else {}

def get_default_config():
    # Retorna la configuración por defecto para la demostración.
    default_bounds = [[0.1, 3.0], [0.01, 1.0], [-3.0, 3.0]] * 3
    return {
        "common_seed": 42,
        "common_bounds": default_bounds,
        "evaluator_params": {
            "target": [1.0, 0.8, 0.6],
            "t_span": (0, 40),
            "dt": 0.5,
            "noise_std": 0.05,
            "bounds": default_bounds,
            "initial_conditions": [0.1, 0.1, 0.1],
            "high_fitness_penalty": 1e6,
            "fitness_penalty_factor": 0.001,
            "min_production_rate": 1e-6,
            "min_degradation_rate": 1e-3,
            "seed": 42,
        },
        "ega_params": {
            "populationSize": 30,
            "generations": 25,
            "crossover_rate": 0.8,
            "mutation_rate": 0.15,
            "elite_size": 3,
            "bounds": default_bounds,
            "alpha_blx": 0.15,
            "mutation_scale": [0.05, 0.05, 0.2] * 3,
            "failure_rate_threshold_increase": 0.3,
            "failure_rate_threshold_decrease": 0.05,
            "timeout_increase_factor": 2.0,
            "timeout_decrease_factor": 0.9,
            "base_timeout": 25.0,
            "max_timeout": 60000.0,
            "timeout": 25.0,
            "processes": max(1, os.cpu_count() - 1),
            "seed": 42,
            "tournament_k": 3,
            "strategy": "uniform",
        },
        "spaghetti_plot": {
            "enabled": True, # Corregido a 'enabled'
            "num_simulations": 50,
            "noise_std_factor": 0.5,
        },
        "snapshot_dir": "snapshots"
    }
    
def check_for_unknown_keys(user_config, default_config, path=""):
    """Advierte sobre claves desconocidas en la configuración del usuario de forma recursiva."""
    for key in user_config:
        full_path = f"{path}.{key}" if path else key
        if key not in default_config:
            warnings.warn(f"Advertencia: Clave desconocida '{full_path}' en la configuración. Podría ser un error tipográfico y será ignorada.")
        elif isinstance(user_config.get(key), dict) and isinstance(default_config.get(key), dict):
            check_for_unknown_keys(user_config[key], default_config[key], path=full_path)

def validate_config(config, default_config, path=""):
    """Valida las claves en la configuración del usuario de forma recursiva."""
    for key, value in default_config.items():
        full_path = f"{path}.{key}" if path else key
        if key not in config:
            # Esto no debería ocurrir si se fusiona primero con los valores por defecto, pero es una buena práctica.
            raise ValueError(f"Clave requerida faltante en la configuración: {full_path}")
        if isinstance(value, dict):
            if not isinstance(config[key], dict):
                raise TypeError (f"El valor para la clave '{full_path}' debe ser un diccionario.")
            validate_config(config[key], value, path=full_path)

def merge_configs(default, user):
    """Fusiona dos diccionarios de configuración de forma recursiva."""
    merged = default.copy()
    for key, value in user.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def main():
    """Función principal que ejecuta la demostración."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()
    
    # Cargar configuración desde el archivo y fusionarla con la configuración por defecto
    user_config = load_config(args.config)
    default_config = get_default_config()

    # Advertir sobre claves desconocidas en el archivo de configuración del usuario
    check_for_unknown_keys(user_config, default_config)

    # Fusionar configuraciones
    config = merge_configs(default_config, user_config)

    # Validar configuración
    validate_config(config, default_config)

    if len(config["ega_params"]["mutation_scale"]) != len(config["evaluator_params"]["bounds"]):
        raise ValueError("mutation_scale debe tener la misma longitud que bounds.")

    # Instanciar el evaluador con los parámetros de la configuración
    evaluator = ToyODEEvaluator(config["evaluator_params"])

    """
    # Imprimir configuraciones
    print("Configuración del EGA:")
    pprint.pprint(config["ega_params"])
    print("Configuración del Evaluador:")
    pprint.pprint(config["evaluator_params"])
    print("Configuración del Spaghetti Plot:")
    pprint.pprint(config["spaghetti_plot"])
    """
    
    # Crear y ejecutar el algoritmo genético
    ega = EGA(config["ega_params"], evaluator=evaluator)
    results = ega.run(snapshot_dir=config["snapshot_dir"], verbose=True)
    
    # Imprimir resultados
    print("Mejor solución encontrada:", results["best"])
    print("Tiempo total (s):", results["total_time_s"])
    
if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TypeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
