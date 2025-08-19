"""
ega_core.py (VERSIÓN CORREGIDA)

Este archivo contiene el núcleo del Algoritmo Genético Elitista (EGA).
Está diseñado para funcionar con individuos que son vectores de números reales y permite
la conexión ("plug-in") de diferentes funciones de evaluación.

CAMBIOS PRINCIPALES EN ESTA VERSIÓN:
- Eliminado el deadlock en multiprocessing
- Simplificada la evaluación paralela
- Mejorado el manejo de timeouts
- Corregidos los métodos de selección y creación de descendencia
"""

import json
import os
import time
import random
import math
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Callable, Dict
import signal

# -----------------------
# Utilidades / configuración
# -----------------------
def init_worker(seed_base: int):
    """Inicializador para los 'workers' (procesos hijos).
    
    Esta función asegura que cada proceso hijo tenga una semilla de aleatoriedad diferente pero
    determinista. Esto es crucial para que los resultados del algoritmo sean reproducibles, 
    incluso cuando se utiliza procesamiento en paralelo.
    
    Args:
        seed_base (int): La semilla base para generar las demás semillas.
    """
    import os, random, numpy as _np
    pid = os.getpid()
    s = (seed_base + pid) % (2**31 - 1)
    random.seed(s)
    _np.random.seed(s)

def safe_round_tuple(arr, digits=6):
    return tuple([round(float(x), digits) for x in arr])

def evaluate_individual_wrapper(args):
    """Función wrapper para evaluación paralela que maneja timeouts."""
    individual_params, evaluator_func, timeout = args
    
    # Configurar timeout usando signal (solo funciona en Unix)
    def timeout_handler(signum, frame):
        raise TimeoutError("Evaluation timeout")
    
    try:
        if hasattr(signal, 'SIGALRM'):  # Solo en sistemas Unix
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        # Evaluar el individuo
        fitness = evaluator_func(individual_params)
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancelar la alarma
        
        return fitness
    except (TimeoutError, ValueError) as e:
        return float('inf')  # Penalizar individuos que fallan
    except Exception as e:
        print(f"Error durante evaluación: {e}")
        return float('inf')

# -----------------------
# Clase Individual
# -----------------------
class Individual:
    """Representa a un individuo en la población del algoritmo genético.
    
    Un individuo contiene un conjunto de parámetros (genes) que representan una posible 
    solución al problema. También almacena su 'fitness', que es una medida de qué tan buena 
    es esa solución.
    """
    def __init__(self, bounds: np.ndarray, init_strategy: str = "uniform"):
        """
        Inicializa un nuevo individuo.

        Args:
            bounds (np.ndarray): Un array que define los límites (mínimo y máximo) para cada 
                                 parámetro del individuo. Tiene la forma (n_params, 2).
            init_strategy (str): La estrategia para inicializar los parámetros. 
                                 Puede ser 'uniform' (aleatorio dentro de los límites) o 
                                 'center' (justo en el medio de los límites).
        """
        self.bounds = np.array(bounds, dtype=float)
        self.n = self.bounds.shape[0]
        if init_strategy == "uniform":
            self.params = self.bounds[:,0] + np.random.rand(self.n) * (self.bounds[:,1] - self.bounds[:,0])
        elif init_strategy == "center":
            self.params = (self.bounds[:,0] + self.bounds[:,1]) / 2.0
        else:
            raise ValueError("Unknown init_strategy")
        self.fitness = None

    def decode(self):
        """Decodifica al individuo, devolviendo sus parámetros.
        
        Este método asegura que los parámetros del individuo estén siempre dentro de los 
        límites definidos. Si algún parámetro se sale de sus límites (por ejemplo, después 
        de una mutación), se "recorta" para que vuelva a estar dentro.
        
        Returns:
            np.ndarray: El vector de parámetros del individuo.
        """
        return np.clip(self.params, self.bounds[:,0], self.bounds[:,1])

    def copy(self):
        ind = Individual(self.bounds)
        ind.params = np.array(self.params, copy=True)
        ind.fitness = self.fitness
        return ind

# -----------------------
# Operadores genéticos
# -----------------------
def blx_alpha_crossover(p1: np.ndarray, p2: np.ndarray, alpha=0.3, bounds=None):
    """Realiza el cruzamiento BLX-alpha entre dos padres para crear un hijo.
    
    Este tipo de cruzamiento está diseñado para variables continuas. Crea un nuevo valor
    para cada gen del hijo tomándolo de un rango extendido alrededor de los valores de los
    padres. El parámetro 'alpha' controla cuánto se puede extender este rango.
    
    Args:
        p1 (np.ndarray): Vector de parámetros del primer padre.
        p2 (np.ndarray): Vector de parámetros del segundo padre.
        alpha (float): Factor de expansión del rango.
        bounds (np.ndarray): Límites globales para los parámetros.
        
    Returns:
        np.ndarray: El vector de parámetros del hijo.
    """
    child_params = np.zeros_like(p1)
    for i in range(len(p1)):
        d = abs(p1[i] - p2[i])
        min_val = min(p1[i], p2[i]) - alpha * d
        max_val = max(p1[i], p2[i]) + alpha * d
        child_params[i] = random.uniform(min_val, max_val)
    if bounds is not None:
        child_params = np.clip(child_params, bounds[:,0], bounds[:,1])
    return child_params

def gaussian_mutation(params: np.ndarray, mutation_rate: float, mutation_scale: np.ndarray, bounds=None):
    """Aplica una mutación gaussiana a un vector de parámetros.
    
    Para cada parámetro (gen), hay una probabilidad ('mutation_rate') de que mute. 
    Si muta, se le suma un pequeño valor aleatorio obtenido de una distribución normal
    (gaussiana). La 'mutation_scale' controla cuán grande puede ser este cambio.
    
    Args:
        params (np.ndarray): Vector de parámetros a mutar.
        mutation_rate (float): Probabilidad de que cada parámetro mute.
        mutation_scale (np.ndarray): La escala (desviación estándar) de la mutación para cada parámetro.
        bounds (np.ndarray): Límites para asegurar que los parámetros no se salgan de su rango.
        
    Returns:
        np.ndarray: El vector de parámetros mutado.
    """
    n = len(params)
    mutated_params = params.copy()
    for i in range(n):
        if random.random() < mutation_rate:
            std = mutation_scale[i]
            # Escala absoluta basada en los límites
            if bounds is not None:
                abs_std = std * (bounds[i,1] - bounds[i,0])
            else:
                abs_std = std
            mutated_params[i] += np.random.normal(0, abs_std)
    if bounds is not None:
        mutated_params = np.clip(mutated_params, bounds[:,0], bounds[:,1])
    return mutated_params

# -----------------------
# EGA core (CORREGIDO)
# -----------------------
class EGA:
    """Clase principal que implementa el Algoritmo Genético Elitista.
    
    Esta clase orquesta todo el proceso evolutivo: inicializa la población, la evalúa,
    selecciona a los padres, crea descendencia a través del cruzamiento y la mutación,
    y repite el proceso durante un número determinado de generaciones.
    """
    def __init__(self, config: Dict, evaluator):
        """
        Inicializa el algoritmo genético.

        Args:
            config (Dict): Un diccionario con todos los parámetros de configuración del 
                           algoritmo (tamaño de población, generaciones, tasas, etc.).
            evaluator: Un objeto que sabe cómo evaluar a un individuo. Debe tener un 
                       método 'evaluate(params)' que devuelva un valor de 'fitness' 
                       (donde un número más bajo es mejor).
        """
        self.config = config
        random.seed(config.get("seed", 42))
        np.random.seed(config.get("seed", 42))
        self.evaluator = evaluator
        self.bounds = np.array(config["bounds"], dtype=float)
        self.pop_size = int(config.get("populationSize", 40))
        self.generations = int(config.get("generations", 60))
        self.crossover_rate = float(config.get("crossover_rate", 0.7))
        self.mutation_rate = float(config.get("mutation_rate", 0.15))
        self.elite_size = int(config.get("elite_size", 2))
        self.alpha_blx = float(config.get("alpha_blx", 0.3))
        self.mutation_scale = np.array(config.get("mutation_scale", [0.05]*self.bounds.shape[0]), dtype=float)
        self.timeout = float(config.get("timeout", 30.0))
        self.processes = int(max(1, min(cpu_count()-1, config.get("processes", cpu_count()-1))))
        self.seed = int(config.get("seed", 42))
        self.tournament_k = int(config.get("tournament_k", 3))
        self.cache = {}  # caching evaluations: key -> fitness
        self.population = [Individual(self.bounds) for _ in range(self.pop_size)]
        self.history = {"min": [], "avg": [], "gen_time": []}

        # Asignación de los operadores genéticos
        self._crossover = partial(blx_alpha_crossover, alpha=self.alpha_blx, bounds=self.bounds)
        self._mutation = partial(gaussian_mutation, 
                                mutation_rate=self.mutation_rate, 
                                mutation_scale=self.mutation_scale, 
                                bounds=self.bounds)

        # No creamos el pool aquí, sino que lo crearemos cuando lo necesitemos

    def _evaluate_population_parallel(self, population_to_eval):
        """Evalúa una población usando procesamiento paralelo (CORREGIDO)."""
        # Preparar individuos que necesitan evaluación
        eval_needed = []
        for ind in population_to_eval:
            if ind.fitness is None:
                key = safe_round_tuple(ind.decode())
                if key in self.cache:
                    ind.fitness = self.cache[key]
                else:
                    eval_needed.append(ind)
        
        if not eval_needed:
            return  # Todos ya están evaluados
        
        # Preparar argumentos para la evaluación paralela
        eval_args = [(ind.decode(), self.evaluator.evaluate, self.timeout) for ind in eval_needed]
        
        # Crear pool temporalmente y evaluar
        with Pool(processes=self.processes, initializer=init_worker, initargs=(self.seed,)) as pool:
            results = pool.map(evaluate_individual_wrapper, eval_args)
        
        # Asignar resultados y actualizar caché
        for ind, fitness in zip(eval_needed, results):
            ind.fitness = fitness
            key = safe_round_tuple(ind.decode())
            self.cache[key] = fitness

    def tournament_selection(self, k=None, num_select=None):
        """Realiza una selección por torneo.

        Para cada selección, se elige un grupo de 'k' individuos al azar y el que tiene
        el mejor fitness (el más bajo) es seleccionado.

        Args:
            k (int): Número de participantes en cada torneo.
            num_select (int): Número total de individuos a seleccionar.

        Returns:
            List[Individual]: La lista de individuos seleccionados.
        """
        if k is None:
            k = self.tournament_k
        if num_select is None:
            num_select = self.pop_size
            
        selected = []
        for _ in range(num_select):
            participants = random.sample(self.population, min(k, len(self.population)))
            winner = min(participants, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
            selected.append(winner.copy())  # Hacemos una copia para evitar modificaciones accidentales
        return selected

    def _create_offspring(self, parents):
        """Crea la descendencia a partir de los padres seleccionados."""
        offspring = []
        num_offspring_needed = self.pop_size - self.elite_size
        
        i = 0
        while len(offspring) < num_offspring_needed:
            # Seleccionar dos padres
            p1 = parents[i % len(parents)]
            p2 = parents[(i + 1) % len(parents)]

            # Crear hijo mediante cruzamiento o copia
            if random.random() < self.crossover_rate:
                child_params = self._crossover(p1.params, p2.params)
            else:
                child_params = p1.params.copy()

            # Crear el individuo hijo
            child = Individual(self.bounds)
            child.params = child_params
            
            # Aplicar mutación
            child.params = self._mutation(child.params)
            
            offspring.append(child)
            i += 1
        
        return offspring

    def run(self, snapshot_dir="snapshots", verbose=True):
        """Ejecuta el bucle principal del algoritmo genético.

        Itera a través de las generaciones, aplicando elitismo, selección, cruzamiento
        y mutación para crear nuevas poblaciones. Guarda snapshots y muestra el progreso.

        Args:
            snapshot_dir (str): Directorio donde se guardarán los resultados y snapshots.
            verbose (bool): Si es True, imprime información del progreso en cada generación.

        Returns:
            dict: Un diccionario con los resultados finales del algoritmo.
        """
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Evaluación inicial
        t0 = time.time()
        print("Evaluando población inicial...")
        self._evaluate_population_parallel(self.population)
        
        gen = 0
        self._record_stats(gen)
        if verbose:
            min_fitness = min([p.fitness for p in self.population if p.fitness is not None])
            avg_fitness = np.mean([p.fitness for p in self.population if p.fitness is not None])
            print(f"[Gen {gen}] min={min_fitness:.6g}; avg={avg_fitness:.6g}")

        for gen in range(1, self.generations + 1):
            start = time.time()

            # 1. Elitismo: preservar los K mejores
            self.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
            elites = [ind.copy() for ind in self.population[:self.elite_size]]

            # 2. Selección de padres
            parents = self.tournament_selection(num_select=self.pop_size - self.elite_size)

            # 3. Creación de descendencia (incluye cruzamiento y mutación)
            offspring = self._create_offspring(parents)

            # 4. Evaluar a los nuevos individuos
            self._evaluate_population_parallel(offspring)

            # 5. Formar la nueva población
            self.population = elites + offspring

            # Registrar estadísticas
            self._record_stats(gen)
            end = time.time()
            self.history["gen_time"].append(end - start)

            # Guardar snapshot
            min_fitness = min([p.fitness for p in self.population if p.fitness is not None])
            avg_fitness = np.mean([p.fitness for p in self.population if p.fitness is not None])
            
            snapshot = {
                "gen": gen,
                "min": float(min_fitness),
                "avg": float(avg_fitness),
                "best_params": self.population[0].params.tolist(),
                "seed": self.seed,
                "config": self.config
            }
            
            with open(os.path.join(snapshot_dir, f"snapshot_gen_{gen}.json"), "w") as fh:
                json.dump(snapshot, fh, indent=2)

            if verbose:
                print(f"[Gen {gen}] min={snapshot['min']:.6g}; avg={snapshot['avg']:.6g}; time={self.history['gen_time'][-1]:.2f}s")

        total_time = time.time() - t0
        
        # Guardar resultado final
        final = {
            "history": self.history,
            "best": {
                "params": self.population[0].params.tolist(),
                "fitness": float(self.population[0].fitness) if self.population[0].fitness is not None else float('inf')
            },
            "config": self.config,
            "total_time_s": total_time
        }
        
        with open(os.path.join(snapshot_dir, "final_result.json"), "w") as fh:
            json.dump(final, fh, indent=2)
        
        return final

    def _record_stats(self, gen):
        """Registra las estadísticas de la población actual (fitness mínimo y promedio).

        Args:
            gen (int): El número de la generación actual.
        """
        valid_fitness = [ind.fitness for ind in self.population if ind.fitness is not None]
        if valid_fitness:
            self.history["min"].append(float(np.min(valid_fitness)))
            self.history["avg"].append(float(np.mean(valid_fitness)))
        else:
            self.history["min"].append(float('inf'))
            self.history["avg"].append(float('inf'))

# EOF