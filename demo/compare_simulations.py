import yaml
import os
import sys
import copy
import numpy as np
from itertools import product
import time

# Añadir la ruta del proyecto al path para poder importar run_demo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.run_demo import main as run_demo_main

def load_config(path):
    """Carga un archivo de configuración en formato YAML."""
    with open(path, "r") as filehandler:
        config = yaml.safe_load(filehandler)
    return config if config is not None else {}

def save_config(path, config):
    """Guarda la configuración en un archivo YAML."""
    with open(path, "w") as filehandler:
        yaml.dump(config, filehandler, default_flow_style=False)

def calculate_target_distance(y_values, target_values):
    """Calcula la distancia promedio entre los valores finales y los valores objetivo."""
    if y_values is None or len(y_values) != len(target_values):
        return float('inf')  # Retorna infinito si hay un problema con los datos
    
    distances = [abs(y - target) for y, target in zip(y_values, target_values)]
    return sum(distances) / len(distances)

def evaluate_simulation(results, target_values, generations, max_generations):
    """Evalúa los resultados de una simulación según criterios biológicos."""
    # Extraer métricas relevantes
    best_fitness = results["best"]["fitness"]
    last_avg_fitness = results["history"]["avg"][-1] if results["history"]["avg"] else float('inf')
    
    # Extraer los valores finales de y (concentraciones de proteínas)
    best_trajectory = results["best"]["y"]
    final_y_values = [traj[-1] for traj in best_trajectory] if best_trajectory else None
    
    # Calcular distancia a los valores objetivo
    target_distance = np.linalg.norm(np.array(final_y_values) - np.array(target_values))
    target_distance = calculate_target_distance(final_y_values, target_values)
    
    # Pesos adaptativos basados en el número de generaciones
    # Menos generaciones: prioriza la eficiencia (encontrar un buen fitness rápido)
    # Más generaciones: prioriza la precisión (acercarse al objetivo)
    efficiency_weight = 1.0 / generations
    precision_weight = generations / max_generations  # Normalizado por el máx. de generaciones
    
    # Puntuación combinada (menor es mejor)
    # Es una función de costo que queremos minimizar.
    score = (best_fitness * efficiency_weight) + \
            (last_avg_fitness * efficiency_weight * 0.5) + \
            (target_distance * precision_weight)
    
    return {
        "best_fitness": best_fitness,
        "last_avg_fitness": last_avg_fitness,
        "target_distance": target_distance,
        "final_y_values": final_y_values,
        "score": score
    }

def optimize_parameters(config_path):
    """Busca la mejor combinación de parámetros para la simulación."""
    original_config = load_config(config_path)
    target_values = original_config["evaluator_params"]["target"]
    
    # Definir los valores a explorar para cada parámetro
    strategies = ["uniform", "center"]
    population_sizes = list(range(10, 101))  # Valores en el rango 10-100
    generations_values = list(range(10, 201))  # Valores en el rango 10-200
    
    print(f"Iniciando optimización de parámetros con {len(strategies)*len(population_sizes)*len(generations_values)} combinaciones")
    print(f"Valores objetivo: {target_values}")
    
    best_params = None
    best_evaluation = None
    best_score = float('inf')
    
    # Guardar resultados para análisis posterior
    all_results = []
    
    # Contador para seguimiento del progreso
    total_combinations = len(strategies) * len(population_sizes) * len(generations_values)
    current_combination = 0
    
    start_time = time.time()
    
    # Explorar todas las combinaciones de parámetros
    for strategy, pop_size, gens in product(strategies, population_sizes, generations_values):
        current_combination += 1
        
        # Mostrar progreso
        elapsed_time = time.time() - start_time
        avg_time_per_combination = elapsed_time / current_combination
        estimated_remaining = avg_time_per_combination * (total_combinations - current_combination)
        
        print(f"\nCombinación {current_combination}/{total_combinations} - "
              f"Tiempo estimado restante: {estimated_remaining/60:.1f} minutos")
        print(f"Parámetros: strategy={strategy}, populationSize={pop_size}, generations={gens}")
        
        # Crear una copia de la configuración original
        config = copy.deepcopy(original_config)
        
        # Modificar los parámetros
        config["ega_params"]["strategy"] = strategy
        config["ega_params"]["populationSize"] = pop_size
        config["ega_params"]["generations"] = gens
        
        # Guardar la configuración modificada
        save_config(config_path, config)
        
        # Ejecutar la simulación
        try:
            simulation_results = run_demo_main()
            
            # Evaluar los resultados
            evaluation = evaluate_simulation(simulation_results, target_values, gens, len(generations_values))
            
            # Guardar los resultados con sus parámetros
            result_entry = {
                "strategy": strategy,
                "populationSize": pop_size,
                "generations": gens,
                "evaluation": evaluation
            }
            all_results.append(result_entry)
            
            # Actualizar el mejor resultado si es necesario
            if evaluation["score"] < best_score:
                best_score = evaluation["score"]
                best_evaluation = evaluation
                best_params = {
                    "strategy": strategy,
                    "populationSize": pop_size,
                    "generations": gens
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
    
    # Análisis por generations
    gens_analysis = {}
    for gens in generations_values:
        gens_results = [result for result in all_results if result["generations"] == gens]
        avg_score = sum(result["evaluation"]["score"] for result in gens_results) / len(gens_results)
        gens_analysis[gens] = avg_score
    
    print("\nRendimiento promedio por número de generaciones:")
    for gens, avg_score in sorted(gens_analysis.items()):
        print(f"Generaciones {gens}: {avg_score:.4f}")
    
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