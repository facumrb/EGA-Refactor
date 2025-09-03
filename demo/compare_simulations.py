import yaml
import os
import sys
import copy
import numpy as np
from itertools import product
import time

# Añadir la ruta del proyecto al path para poder importar run_demo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.run_demo import run_simulation_with_config as run

def load_config(path):
    """Carga un archivo de configuración en formato YAML."""
    with open(path, "r") as filehandler:
        config = yaml.safe_load(filehandler)
    return config if config is not None else {}

def save_config(path, config):
    """Guarda la configuración en un archivo YAML."""
    with open(path, "w") as filehandler:
        yaml.dump(config, filehandler, default_flow_style=False)

def optimize_parameters(config_path):
    """Busca la mejor combinación de parámetros para la simulación."""
    original_config = load_config(config_path)
    target_values = original_config["evaluator_params"]["target"]
    
    # Definir los valores a explorar para cada parámetro
    strategies = ["uniform", "center"]
    population_sizes = list(range(10, 101))  # Valores en el rango 10-100
    
    print(f"Iniciando optimización de parámetros con {len(strategies)*len(population_sizes)} combinaciones")
    print(f"Valores objetivo: {target_values}")
    
    best_params = None
    best_evaluation = None
    best_score = float('inf')
    
    # Guardar resultados para análisis posterior
    all_results = []
    
    # Contador para seguimiento del progreso
    total_combinations = len(strategies) * len(population_sizes)
    current_combination = 0
    
    # Explorar todas las combinaciones de parámetros
    for strategy, pop_size in product(strategies, population_sizes):
        current_combination += 1
        
        print(f"\nCombinación {current_combination}/{total_combinations}")
        print(f"Parámetros: strategy={strategy}, populationSize={pop_size}")
        
        # Crear una copia de la configuración original
        config = copy.deepcopy(original_config)
        
        # Modificar los parámetros
        config["ega_params"]["strategy"] = strategy
        config["ega_params"]["populationSize"] = pop_size
        
        # Guardar la configuración modificada
        save_config(config_path, config)
        
        # Ejecutar la simulación
        try:
            # Guardar los resultados con sus parámetros
            pop_results_entry = run(config_path, compare=True)
            
            for gen_results in pop_results_entry:
                all_results.append(gen_results)
                # Actualizar el mejor resultado si es necesario
                if gen_results["evaluation"]["score"] < best_score:
                    best_score = gen_results["evaluation"]["score"]
                    best_evaluation = gen_results["evaluation"]
                    best_params = {
                        "strategy": strategy,
                        "populationSize": pop_size,
                        "generations": gen_results["generations"]
                    }
            
        except Exception as error:
            print(f"Error en la simulación: {error}")
    
    # Restaurar la configuración original
    save_config(config_path, original_config)
    
    # Mostrar los mejores resultados
    print("\n" + "=" * 50)
    print("MEJORES PARÁMETROS ENCONTRADOS:")
    print(f"Estrategia: {best_params['strategy']}")
    print(f"Tamaño de la población: {best_params['populationSize']}")
    print(f"Generaciones: {best_params['generations']}")
    print("\nRESULTADOS:")
    print(f"Fitness: {best_evaluation['best_fitness']}")
    print(f"Último fitness promedio: {best_evaluation['last_avg_fitness']}")
    print(f"Distancia a valores objetivo: {best_evaluation['target_distance']}")
    print(f"Valores finales: {best_evaluation['final_y_values']}")
    print(f"Valores objetivo: {target_values}")
    print(f"Puntuación final: {best_evaluation['score']}")
    
    # Análisis de tendencias en los resultados
    print("\n" + "=" * 50)
    print("ANÁLISIS DE TENDENCIAS:")
    
    # Ordenar resultados por puntuación
    sorted_results = sorted(all_results, key=lambda x: x["evaluation"]["score"])
    top_results = sorted_results[:5]  # Top 5 mejores resultados
    
    print("\nTop 5 mejores combinaciones:")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. Estrategia: {result['strategy']}, "
              f"Tamaño de población: {result['populationSize']}, "
              f"Generaciones: {result['generations']}, "
              f"Score: {result['evaluation']['score']:.4f}")
    
    # Análisis por strategy
    strategies_analysis = {}
    for strategy in strategies:
        strategy_results = [result for result in all_results if result["strategy"] == strategy]
        avg_score = sum(result["evaluation"]["score"] for result in strategy_results) / len(strategy_results)
        strategies_analysis[strategy] = avg_score
    
    print("\nRendimiento promedio por estrategia:")
    for strategy, avg_score in strategies_analysis.items():
        print(f"Estrategia '{strategy}': {avg_score:.4f}")
    
    # Análisis por populationSize
    pop_size_analysis = {}
    for pop_size in population_sizes:
        pop_results = [result for result in all_results if result["populationSize"] == pop_size]
        avg_score = sum(result["evaluation"]["score"] for result in pop_results) / len(pop_results)
        pop_size_analysis[pop_size] = avg_score
    
    print("\nRendimiento promedio por tamaño de población:")
    for pop_size, avg_score in sorted(pop_size_analysis.items()):
        print(f"Tamaño de población {pop_size}: {avg_score:.4f}")
    
    # Guardar los resultados completos para análisis posterior
    results_file = os.path.join(os.path.dirname(config_path), "optimization_results.yaml")
    with open(results_file, "w") as filehandler:
        yaml.dump({
            "best_params": best_params,
            "best_evaluation": best_evaluation,
            "all_results": all_results,
            "strategies_analysis": strategies_analysis,
            "pop_size_analysis": pop_size_analysis,
            "gens_analysis": gens_analysis
        }, filehandler, default_flow_style=False)
    
    print(f"\nResultados completos guardados en {results_file}")
    
    # Aplicar los mejores parámetros encontrados a la configuración
    best_config = copy.deepcopy(original_config)
    best_config["ega_params"]["strategy"] = best_params["strategy"]
    best_config["ega_params"]["populationSize"] = best_params["populationSize"]
    best_config["ega_params"]["generations"] = best_params["generations"]
    
    best_config_file = os.path.join(os.path.dirname(config_path), "best_config.yaml")
    save_config(best_config_file, best_config)
    print(f"Mejor configuración guardada en {best_config_file}")
    
    return best_params, best_evaluation

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    optimize_parameters(config_path)