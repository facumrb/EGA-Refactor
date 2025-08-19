"""
plot_results.py

Script para visualizar los resultados del algoritmo genético.
Crea gráficos que muestran la evolución del fitness y compara
la simulación del mejor individuo con el objetivo.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from evaluator_toy import ToyODEEvaluator
import sys
import os

def load_results(results_path="snapshots/final_result.json"):
    """Carga los resultados del algoritmo genético."""
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {results_path}")
        print("Asegúrate de haber ejecutado run_demo.py primero.")
        return None

def plot_fitness_evolution(results):
    """Grafica la evolución del fitness a lo largo de las generaciones."""
    history = results['history']
    generations = range(len(history['min']))
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Evolución del fitness
    plt.subplot(1, 2, 1)
    plt.plot(generations, history['min'], 'b-', label='Mejor Fitness', linewidth=2)
    plt.plot(generations, history['avg'], 'r--', label='Fitness Promedio', alpha=0.7)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Tiempo por generación
    plt.subplot(1, 2, 2)
    if len(history['gen_time']) > 0:
        plt.plot(range(1, len(history['gen_time']) + 1), history['gen_time'], 'g-', alpha=0.7)
        plt.xlabel('Generación')
        plt.ylabel('Tiempo (segundos)')
        plt.title('Tiempo de Ejecución por Generación')
        plt.grid(True, alpha=0.3)
        
        avg_time = np.mean(history['gen_time'])
        plt.axhline(y=avg_time, color='r', linestyle='--', 
                   label=f'Promedio: {avg_time:.2f}s')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('snapshots/fitness_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_best_simulation(results):
    """Simula y grafica el comportamiento del mejor individuo encontrado."""
    best_params = results['best']['params']
    config = results['config']
    
    # Crear evaluador con la misma configuración
    evaluator = ToyODEEvaluator(
        target=np.array(config['target']),
        t_span=tuple(config['t_span']),
        dt=config['dt'],
        noise_std=config.get('noise_std', 0.0)
    )
    
    # Simular el mejor individuo
    y_final, sol = evaluator.simulate(best_params)
    
    if sol is None:
        print("Error: No se pudo simular el mejor individuo")
        return
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Evolución temporal completa
    plt.subplot(2, 2, 1)
    for i in range(3):
        plt.plot(sol.t, sol.y[i, :], label=f'Factor {i+1}', linewidth=2)
    
    # Líneas horizontales para el objetivo
    target = np.array(config['target'])
    for i in range(3):
        plt.axhline(y=target[i], color=f'C{i}', linestyle='--', alpha=0.5,
                   label=f'Objetivo {i+1}: {target[i]}')
    
    plt.xlabel('Tiempo')
    plt.ylabel('Concentración')
    plt.title('Evolución Temporal - Mejor Solución')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Comparación final vs objetivo
    plt.subplot(2, 2, 2)
    x_pos = np.arange(3)
    width = 0.35
    
    plt.bar(x_pos - width/2, y_final, width, label='Resultado Final', alpha=0.8)
    plt.bar(x_pos + width/2, target, width, label='Objetivo', alpha=0.8)
    
    plt.xlabel('Factor de Transcripción')
    plt.ylabel('Concentración Final')
    plt.title('Comparación: Resultado vs Objetivo')
    plt.xticks(x_pos, [f'Factor {i+1}' for i in range(3)])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Parámetros del mejor individuo
    plt.subplot(2, 2, 3)
    param_names = []
    for i in range(3):
        param_names.extend([f'TF{i+1}_prod', f'TF{i+1}_deg', f'TF{i+1}_inter'])
    
    bars = plt.bar(range(len(best_params)), best_params)
    plt.xlabel('Parámetro')
    plt.ylabel('Valor')
    plt.title('Parámetros del Mejor Individuo')
    plt.xticks(range(len(best_params)), param_names, rotation=45)
    
    # Colorear barras según el tipo de parámetro
    colors = ['skyblue', 'lightcoral', 'lightgreen'] * 3
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Métricas de calidad
    plt.subplot(2, 2, 4)
    distance = np.linalg.norm(y_final - target)
    relative_errors = np.abs(y_final - target) / target * 100
    
    metrics = ['Distancia L2', 'Error TF1 (%)', 'Error TF2 (%)', 'Error TF3 (%)']
    values = [distance] + relative_errors.tolist()
    
    bars = plt.bar(range(len(metrics)), values)
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    plt.title('Métricas de Calidad')
    plt.xticks(range(len(metrics)), metrics, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('snapshots/best_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Imprimir resumen numérico
    print("\n" + "="*60)
    print("RESUMEN DE LA MEJOR SOLUCIÓN ENCONTRADA")
    print("="*60)
    print(f"Fitness final: {results['best']['fitness']:.6f}")
    print(f"Distancia L2 al objetivo: {distance:.6f}")
    print(f"Tiempo total de optimización: {results['total_time_s']:.2f} segundos")
    print("\nResultados finales vs Objetivo:")
    for i in range(3):
        error_pct = relative_errors[i]
        print(f"  Factor {i+1}: {y_final[i]:.4f} (objetivo: {target[i]:.4f}, error: {error_pct:.2f}%)")
    
    print("\nMejores parámetros encontrados:")
    for i, (name, value) in enumerate(zip(param_names, best_params)):
        print(f"  {name}: {value:.4f}")

def main():
    """Función principal del script de visualización."""
    print("Cargando resultados del algoritmo genético...")
    
    # Permitir especificar ruta personalizada
    results_path = "snapshots/final_result.json"
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    
    results = load_results(results_path)
    if results is None:
        return
    
    print("Creando gráficos...")
    
    # Crear directorio de salida si no existe
    os.makedirs('snapshots', exist_ok=True)
    
    # Generar gráficos
    plot_fitness_evolution(results)
    plot_best_simulation(results)
    
    print("\nGráficos guardados en:")
    print("  - snapshots/fitness_evolution.png")
    print("  - snapshots/best_simulation.png")
    print("\n¡Visualización completada!")

if __name__ == "__main__":
    # Verificar si matplotlib está disponible
    try:
        import matplotlib.pyplot as plt
        main()
    except ImportError:
        print("Error: matplotlib no está instalado.")
        print("Instálalo con: pip install matplotlib")
        print("Luego ejecuta este script nuevamente.")