"""
ega_core.py

Este archivo contiene el núcleo del Algoritmo Genético Elitista (EGA).
Está diseñado para funcionar con individuos que son vectores de números reales y permite
la conexión ("plug-in") de diferentes funciones de evaluación.

Características Principales:
- Individuo: Representado como un vector de números reales, cada uno con sus propios límites (bounds).
- Selección: Implementa la selección por torneo (por defecto) y también la selección por ruleta.
- Cruzamiento: Utiliza el método BLX-alpha, adecuado para variables continuas (números reales).
- Mutación: Aplica una mutación gaussiana adaptativa.
- Elitismo: Conserva a los 'K' mejores individuos de una generación a la siguiente sin cambios.
- Evaluación Paralela: Usa un "pool" de procesos para evaluar a los individuos simultáneamente, 
  lo que acelera el cálculo. Incluye un tiempo máximo (timeout) por evaluación.
- Caché de Evaluaciones: Almacena los resultados de evaluaciones ya realizadas para no repetir cálculos.
- Snapshots: Guarda el estado del algoritmo (en formato JSON) en cada generación para poder analizar su progreso.
"""

import json
import os
import time
import random
import math
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Dict
from evaluator_toy import ToyODEEvaluator

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

def safe_round_tuple(individual_arr, digits=6):
    """
    Crea una clave única y utilizable para el diccionario de caché ( self.cache ). 
    Los diccionarios en Python necesitan que sus claves no puedan cambiar (sean inmutables). 
    Como los params de un individuo son np.ndarray y pueden cambiar, no se pueden usar directamente como clave. 
    Esta función los convierte en una tupla, que es inmutable y puede ser usada como clave.
    """
    return tuple([round(float(gen), digits) for gen in individual_arr])

def _evaluator_wrapper(evaluator_and_individual):
    """Envoltorio para llamar al método evaluate del evaluador. Puede ser serializado."""
    # Desempaqueta los argumentos
    evaluator, individual = evaluator_and_individual
    # Llama al método evaluate del evaluador
    return evaluator.evaluate(individual)

# init_worker y _evaluator_wrapper son funciones auxiliares que se utilizan para 
# inicializar los procesos hijos y evaluar los individuos en paralelo.
# safe_round_tuple es una función auxiliar que se utiliza para convertir los params de un individuo en una tupla.
# Esta tupla se usa como clave en el diccionario de caché.
# El diccionario de caché se utiliza para almacenar los resultados de las evaluaciones previas,
# lo que permite ahorrar tiempo en evaluaciones repetidas.

# -----------------------
# Clase Individual
# -----------------------
class Individual:
    """Representa a un individuo en la población del algoritmo genético.
    
    Propósito : Es el objeto fundamental. 
    No es más que un contenedor para un np.ndarray (un vector de números) llamado params,
    que representa una única solución candidata al problema. 
    También guarda su fitness una vez calculado.

    Biología : Un Individual es análogo a un organismo. 
    Sus params son su "genotipo", una receta numérica. 
    Su fitness es una medida de su éxito en el "ambiente" simulado (el fenotipo).

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

        bounds define el espacio de búsqueda de soluciones viables, es decir, las restricciones 
        físicas o biológicas de los parámetros del sistema.
        self.params es el genotipo del individuo, la secuencia específica de valores que lo define.
        self.fitness es la aptitud biológica, una medida cuantitativa de cuán bien se adapta ese 
        genotipo al entorno (al problema a resolver).
        """
        
        self.bounds = np.array(bounds, dtype=float) 
        # bounds : Representan límites fisiológicos
        self.num_params = self.bounds.shape[0] # Número de parámetros
        # .shape[0]: Obtiene la cantidad de filas del array (cada fila = 1 parámetro)
        # self.num_params : Almacena el total de parámetros a optimizar

        # Inicialización aleatoria dentro de los límites fisiológicos
        # self.bounds[:,1] selecciona la columna de límites superiores.
        # self.bounds[:,0] selecciona la columna de límites inferiores.
        if init_strategy == "uniform":
            self.params = self.bounds[:,0] + np.random.rand(self.num_params) * (self.bounds[:,1] - self.bounds[:,0])
            # La resta calcula el rango o la amplitud permitida para cada parámetro (ej., [max - min] ).
            # np.random.rand(self.num_params) genera números aleatorios entre 0 y 1 para cada parámetro.
            # Estos números aleatorios se multiplican por el rango y se suman al límite inferior para obtener
            # un valor aleatorio dentro de los límites definidos para cada parámetro.
            # Esto simula la variabilidad natural en la formación de genotipos
            # Cada parámetro se inicializa con un valor aleatorio entre su límite mínimo y máximo.
            """
            En un sistema biológico real, los parámetros como las tasas de reacción enzimática o las concentraciones 
            de proteínas no son idénticos en cada célula u organismo. Fluctúan dentro de un rango fisiológicamente viable. 
            Al inicializar los parámetros de forma aleatoria dentro de sus bounds (límites), el algoritmo comienza 
            explorando una solución que representa a un individuo aleatorio y viable de una población heterogénea.
            """
        elif init_strategy == "center":
            self.params = (self.bounds[:,0] + self.bounds[:,1]) / 2.0
            # Suma el límite inferior y superior de cada parámetro.
            # Divide la suma por dos, calculando el punto medio exacto del rango permitido para cada parámetro.
            # Esto asegura que la población se inicialice en un punto central válido dentro de los límites.
            """
            Esta estrategia representa un estado de homeostasis o equilibrio ideal. Inicializa el sistema en un estado 
            promedio, sin sesgos ni variaciones aleatorias. Es útil cuando se quiere partir de una configuración 
            teóricamente balanceada o cuando se sospecha que la solución óptima se encuentra cerca del centro del espacio 
            de búsqueda. Simula un individuo "promedio" o arquetípico del sistema biológico que se está modelando.
            """
        else:
            raise ValueError("Unknown init_strategy")
        # fitness: Representa la aptitud biológica del individuo
        # Inicialmente se establece en None, ya que el individuo no ha sido evaluado todavía.
        self.fitness = None
        self.solution_times = None
        self.trajectory = None

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
        """
        Crea una instancia completamente nueva de la clase Individual.
        Esto es útil cuando se quiere modificar un individuo sin afectar al original.
        """
        individual_copy = Individual(self.bounds)
        individual_copy.params = np.array(self.params, copy=True)
        individual_copy.fitness = self.fitness
        return individual_copy

# -----------------------
# Operadores
# -----------------------
"""
Propósito: Son las funciones que crean nueva diversidad genética. 
Toman uno o dos individuos "padres" y generan un nuevo individuo "hijo" 
con una combinación o variación de sus parámetros.
"""

def blx_alpha_crossover(parent1: np.ndarray, parent2: np.ndarray, blx_alpha=0.3, bounds=None):

    """Realiza el cruzamiento BLX-alpha entre dos padres para crear un hijo.
    
    Este tipo de cruzamiento está diseñado para variables continuas. Crea un nuevo valor
    para cada gen del hijo tomándolo de un rango extendido alrededor de los valores de los
    padres. 
    El parámetro 'alpha' controla cuánto se puede extender este rango:
    si alpha = 0 , el hijo siempre tendrá valores entre los de sus padres. 
    Si alpha > 0 , se permite la extrapolación, es decir, que el hijo explore 
    valores ligeramente fuera del rango de sus padres, lo que puede acelerar la 
    búsqueda de soluciones novedosas. Es un operador de cruce diseñado para 
    variables continuas (números reales).
    
    Args:
        parent1 (np.ndarray): Vector de parámetros del primer padre.
        parent2 (np.ndarray): Vector de parámetros del segundo padre.
        alpha (float): Factor de expansión del rango.
        bounds (np.ndarray): Límites globales para los parámetros.
        
    Returns:
        np.ndarray: El vector de parámetros del hijo.
    """
    num_genes = len(parent1)
    child = np.zeros(num_genes, dtype=float)
    for gen in range(num_genes):
        parent_min = min(parent1[gen], parent2[gen])
        parent_max = max(parent1[gen], parent2[gen])
        parent_range = parent_max - parent_min
        sample_min = parent_min - blx_alpha * parent_range
        sample_max = parent_max + blx_alpha * parent_range
        """
        El parámetro BLX-alpha controla cuánto se permite extrapolar 
        fuera del rango parental (biología: transgresión).
        """
        # Respect global bounds if provided
        if bounds is not None:
            sample_min = max(sample_min, bounds[gen,0])
            sample_max = min(sample_max, bounds[gen,1])
        child[gen] = random.uniform(sample_min, sample_max)
    """
    Simula un fenómeno conocido como transgresión, donde un descendiente puede exhibir 
    un fenotipo más extremo que cualquiera de sus padres. Por ejemplo, dos padres de 
    estatura media pueden tener un hijo más alto o más bajo que ambos. Al permitir que 
    el hijo explore un poco más allá del espacio genético de sus padres ( alpha > 0 ), 
    el algoritmo puede descubrir soluciones novedosas más rápidamente.
    """
    if bounds is not None:
        child = np.clip(child, bounds[:,0], bounds[:,1])
        """
        np.clip(..., bounds[:,0], bounds[:,1]) es crucial: 
        se aseguran de que, incluso después de un cruzamiento o mutación "extremos", 
        los parámetros del nuevo individuo nunca violen los límites biológicamente plausibles 
        que definimos en bounds.
        """
    return child

def gaussian_mutation(individual_params: np.ndarray, mutation_rate: float, mutation_scale: np.ndarray, bounds=None):
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
    elements_num = len(individual_params)
    for gen in range(elements_num):
        if random.random() < mutation_rate:
            standard_deviation = mutation_scale[gen]
            # La desviación estándar se escala para que sea relativa al rango de valores
            # permitidos para ese gen. Esto asegura que la mutación no sea demasiado grande
            # para un gen particular.
            abs_standard_deviation = standard_deviation * (bounds[gen,1] - bounds[gen,0]) if bounds is not None else standard_deviation
            # Se crea una mutación aleatoria para el gen actual.
            mutation = np.random.normal(0, abs_standard_deviation)
            individual_params[gen] += mutation
    """
    Crea un vector de ruido aleatorio del mismo tamaño, extraído de una distribución 
    Gaussiana (Normal) con media 0 y desviación estándar igual a scale.
    Suma este vector de ruido al vector de parámetros original.
    """
    if bounds is not None:
        individual_params = np.clip(individual_params, bounds[:,0], bounds[:,1])
        """
        np.clip(..., bounds[:,0], bounds[:,1]) es crucial: 
        se aseguran de que, incluso después de un cruzamiento o mutación "extremos", 
        los parámetros del nuevo individuo nunca violen los límites biológicamente plausibles 
        que definimos en bounds.
        """
    return individual_params

# -----------------------
# EGA core
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
        Crea la población inicial de individuos aleatorios (dentro de los bounds definidos).

        Args:
            config (Dict): Un diccionario con todos los parámetros de configuración del 
                           algoritmo (tamaño de población, generaciones, tasas, etc.).
            evaluator: Un objeto que sabe cómo evaluar a un individuo. Debe tener un 
                       método 'evaluate(params)' que devuelva un valor de 'fitness' 
                       (donde un número más bajo es mejor).
        """
        self.config = dict(config)
        self.evaluator = evaluator
        self.pop_size = int(config.get("populationSize", 30))
        self.generations = int(config.get("generations", 25))
        self.crossover_rate = float(config.get("crossover_rate", 0.8))
        self.mutation_rate = float(config.get("mutation_rate", 0.15))
        self.elite_size = int(config.get("elite_size", 3))
        self.bounds = np.array(config["bounds"], dtype=float)
        self.alpha_blx = float(config.get("alpha_blx", 0.15))
        self.mutation_scale = np.array(config.get("mutation_scale", [0.05,0.05,0.2, 0.05,0.05,0.2, 0.05,0.05,0.2]), dtype=float)
        self.failure_rate_threshold_increase = float(config.get('failure_rate_threshold_increase', 0.3))
        self.failure_rate_threshold_decrease = float(config.get('failure_rate_threshold_decrease', 0.05))
        self.timeout_increase_factor = float(config.get('timeout_increase_factor', 2.0))
        self.timeout_decrease_factor = float(config.get('timeout_decrease_factor', 0.9))
        self.base_timeout = float(config.get("base_timeout", 25.0))
        self.max_timeout = float(config.get("max_timeout", 300.0))
        self.timeout = float(config.get("timeout", 25.0))
        self.processes = int(max(1, min(cpu_count()-1, config.get("processes", cpu_count()-1))))
        self.seed = int(config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.tournament_k = int(config.get("tournament_k", 3))
        self.strategy = config.get("strategy", "uniform")
        
        self.cache = {}  # caching evaluations: key -> fitness
        """self.previous_avg_fitness = None"""
        self.population = [Individual(self.bounds, self.strategy) for _ in range(self.pop_size)]

        self.history = {"min": [], "avg": [], "gen_time": []}
        # Pool
        self.pool = Pool(processes=self.processes, initializer=init_worker, initargs=(self.seed,))
        # El sistema operativo lanza un número de procesos Python nuevos ( self.processes ). 
        # Cada uno de estos procesos es un "trabajador" casi idéntico al proceso principal.
        # init_worker configura semillas aleatorias. initargs es una tupla con los argumentos
        # que se pasarán a init_worker. En este caso, la semilla. La coma en (self.seed,) 
        # es necesaria para crear tupla de un solo elemento.

    def _evaluate_population(self, population_to_eval=None):
        """
        Evalúa una lista de individuos. Si no se especifica, evalúa la población entera.
        Usa multiprocessing.Pool para enviar a cada individuo a un núcleo de CPU diferente 
        para ser evaluado por el evaluator_toy.py. 
        Esto acelera drásticamente el proceso, que es el cuello de botella computacional.
        Args:
            population_to_eval (List[Individual], optional): La población a evaluar. Si no se 
                                                      especifica, se usa la población actual 
                                                      del objeto EGA. Defaults to None.

        Returns:
            List[float]: Una lista con los valores de fitness de cada individuo.
        """
        if population_to_eval is None:
            population_to_eval = self.population

        # Prepara una lista de individuos que necesitan evaluación y maneja la caché
        eval_needed_individuals = []
        tasks_for_evaluator = []
        for individual in population_to_eval:
            """
            Si un individuo ya tiene un valor de fitness (por ejemplo, porque es un "élite" 
            de la generación anterior que se conservó), no hay nada que hacer y se salta al 
            siguiente individuo del bucle.
            """
            if individual.fitness is None:
                key_for_Dictionary = safe_round_tuple(individual.decode())
                # Se genera una clave única para el genotipo del individuo.
                if key_for_Dictionary in self.cache:
                    individual.fitness = self.cache[key_for_Dictionary]
                    """
                    Si la clave existe, significa que este genotipo ya fue evaluado. Se recupera el 
                    valor de fitness del caché y se le asigna al individuo actual, ahorrando una 
                    evaluación completa.
                    """
                else:
                    eval_needed_individuals.append(individual)
                    tasks_for_evaluator.append((self.evaluator, individual.decode()))

        if not eval_needed_individuals:
            return

        fitness_results = []
        if tasks_for_evaluator:
            # Ejecuta las evaluaciones en paralelo usando el wrapper
            # Ejecuta las evaluaciones en paralelo usando el wrapper
            try:
                async_fitness_solution = self.pool.map_async(_evaluator_wrapper, tasks_for_evaluator)
                fitness_solution_results = async_fitness_solution.get(timeout=self.timeout)
                # Es un cuello de botella:
                # es donde el programa pasa la mayor parte del tiempo. 
                # Cualquier optimización en evaluator_toy.py tiene un impacto directo y masivo 
                # en el tiempo total de ejecución.
                # Asigna los resultados de fitness a los individuos correspondientes y actualiza la caché
                fitness_results, solution_results = zip(*fitness_solution_results)
                for individual, fitness, solution in zip(eval_needed_individuals, fitness_results, solution_results):
                    individual.fitness = float(fitness) if fitness is not None else float('inf')
                    individual.trajectory = solution.y
                    individual.solution_times = solution.t
                    # Si el fitness es infinito, se asume que la evaluación falló.
                    # En este caso, se asigna un valor alto al fitness para que el individuo sea menos apto.
                    key_for_Dictionary = safe_round_tuple(individual.decode())
                    # Se genera una clave única para el genotipo del individuo.
                    self.cache[key_for_Dictionary] = individual.fitness
            except Exception as error:
                print(f"[EGA Error] Evaluación fallida: {error}")
                for i, individual in enumerate(eval_needed_individuals):
                    individual.fitness = float('inf')
                    key = safe_round_tuple(individual.decode())
                    self.cache[key] = individual.fitness
        
    def _select_parents(self, tournament_k):
        """Selecciona a los padres para la siguiente generación usando selección por torneo.
        El torneo  

        Args:
            tournament_k (int): Número de participantes en cada torneo.

        Returns:
            List[Individual]: La lista de individuos seleccionados.
        """
        # num_select indica cuántos padres seleccionar para generar descendientes, asegurando que 
        # la nueva población tenga exactamente self.pop_size individuos.
        return self.tournament_selection(tournament_k, num_select=self.pop_size - self.elite_size)

    
    def _create_offspring(self, parents):
        """Crea la descendencia a partir de los padres seleccionados."""
        # offspring es una lista que almacenará los nuevos individuos (descendientes) generados a partir de los padres seleccionados.
        # Representa la nueva generación que se creará mediante cruce o clonación.
        offspring = []
        i = 0
        # self.pop_size - self.elite_size representa el número de descendientes necesarios para completar la población,
        # excluyendo los espacios reservados para los individuos élite que se copian directamente.
        while len(offspring) < self.pop_size - self.elite_size:
            # En cada iteración, selecciona dos padres consecutivos (parent1 y parent2).
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Aplica cruce con probabilidad self.crossover_rate o clona parent1, crea un nuevo individuo con los parámetros resultantes.
            if random.random() < self.crossover_rate:
                child_params = blx_alpha_crossover(parent1.params, parent2.params, self.alpha_blx)
            else:
                child_params = parent1.params.copy()

            child = Individual(self.bounds)
            child.params = child_params
            # Lo añade a offspring, avanza el índice i en 2.
            offspring.append(child)
            i = (i + 2)
            # Si i excede el límite de parents, reinicia i a 0 para reutilizar padres.
            if i >= len(parents) - 1: i = 0
        return offspring

    def _apply_mutation(self, offspring):
        """Aplica mutación a la descendencia."""
        for child in offspring:
            child.params = gaussian_mutation(child.params, self.mutation_rate, self.mutation_scale, self.bounds)



    # --------------------------------
    # Métodos de evaluación de fitness
    # --------------------------------
    """
    def _eval_single(self, individual: Individual):
        Evalúa a un único individuo, utilizando la caché y gestionando timeouts.

        Args:
            individual (Individual): El individuo a evaluar.

        Returns:
            float: El valor de fitness del individuo.
        
        key = safe_round_tuple(individual.decode())
        if key in self.cache:
            return self.cache[key]
        # apply evaluator with timeout via pool
        try:
            # apply async and get with timeout
            async_res = self.pool.apply_async(self.evaluator.evaluate, (individual.decode(),))
            fitness = async_res.get(timeout=self.timeout)
        except (TimeoutError, ValueError) as e:
            print(f"Evaluation failed for individual {individual.params}: {e}")
            fitness = float('inf')  # Penalize failing individuals
        except Exception as e:
            print(f"An unexpected error occurred during evaluation: {e}")
            fitness = float('inf') # Broad exception for other cases
        # store in cache
        self.cache[key] = float(fitness)
        return float(fitness)
    """	

    # ---------------------
    # Métodos de selección
    # ---------------------
    def tournament_selection(self, tournament_k=3, num_select=None):
        """Realiza una selección por torneo.

        Para cada selección, se elige un grupo de 'k' individuos al azar y el que tiene
        el mejor fitness (el más bajo) es seleccionado.

        Args:
            tournament_k (int): Número de participantes en cada torneo.
            num_select (int): Número total de individuos a seleccionar.

        Returns:
            List[Individual]: La lista de individuos seleccionados.
        """
        if num_select is None:
            num_select = self.pop_size
        selected = []
        for _ in range(num_select):
            participants = random.sample(self.population, tournament_k)
            # Selecciona aleatoriamente tournament_k individuos de la 
            # población actual ( self.population ) sin reemplazo.
            winner = min(participants, key=lambda x: x.fitness)
            # Encuentra el individuo con el menor valor de fitness entre los participantes.
            selected.append(winner)
        return selected

    def roulette_selection(self, num_select=None):
        """Realiza una selección por ruleta.

        La probabilidad de que un individuo sea seleccionado es proporcional a su 'score',
        que se calcula a partir de su fitness (mejor fitness = mayor score).

        Args:
            num_select (int): El número total de individuos a seleccionar.

        Returns:
            List[Individual]: La lista de individuos seleccionados.
        """
        # fitness lower is better; convert to positive scores
        if num_select is None:
            num_select = self.pop_size
        fitnesses = np.array([ind.fitness for ind in self.population], dtype=float)
        # guard contre inf/large
        finite_mask = np.isfinite(fitnesses)
        fitnesses = np.where(finite_mask, fitnesses, np.max(fitnesses[finite_mask]) * 10 + 1)
        scores = 1.0 / (1.0 + fitnesses)  # higher is better
        probs = scores / np.sum(scores)
        idx = np.random.choice(len(self.population), size=num_select, p=probs)
        return [self.population[i] for i in idx]
    
    def _record_stats(self):
        """Registra las estadísticas de la población actual (fitness mínimo y promedio).

        Args:
            gen (int): El número de la generación actual.
        """
        fitness = np.array([individual.fitness for individual in self.population], dtype=float)
        self.history["min"].append(float(np.min(fitness)))
        self.history["avg"].append(float(np.mean(fitness)))

    # --------------------------
    # Bucle principal del algoritmo
    # --------------------------
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
        # Carga en caché de los individuos y sus fitness en base a la primera población:
        self._evaluate_population(self.population)
        # Guarda datos estadísticos en history:
        self._record_stats() # Primer registro de estadísticas
        gen = 0
        # Muestra datos de la primera población:
        if verbose:
            min_fitness = min([individual.fitness for individual in self.population])
            avg_fitness = np.mean([individual.fitness for individual in self.population])
            print(f"[Gen {gen}] min={min_fitness:.6g}; avg={avg_fitness:.6g}")
        
        for gen in range(1, self.generations + 1):
            # Comienza el tiempo de la simulación
            start = time.time()

            # 1. Elitismo: preservar los K mejores
            self.population.sort(key=lambda x: x.fitness)
            # Reorganiza los elementos de self.population en orden ascendente (mejor fitness primero) basado en el valor del atributo x.fitness de cada individuo
            elites = [individual.copy() for individual in self.population[:self.elite_size]]
            # Copia los mejores individuos (elites) en una nueva lista llamada elites.

            # 2. Selección de padres
            parents = self._select_parents(self.tournament_k)

            # 3. Creación de descendencia (cruzamiento)
            offspring = self._create_offspring(parents)

            # 4. Aplicar mutación a la descendencia
            self._apply_mutation(offspring)

            # 5. Evaluar a los nuevos individuos
            self._evaluate_population(offspring) # Evalúa solo a los nuevos

            # 6. Formar la nueva población
            self.population = elites + offspring

            # 7. Registrar estadísticas y tiempo de generación
            self._record_stats() # Siguiente registro de estadísticas
            end = time.time()
            self.history["gen_time"].append(end - start)

            # snapshot
            snapshot = {
                "gen": gen,
                "min": float(min([individual.fitness for individual in self.population])),
                "avg": float(np.mean([individual.fitness for individual in self.population])),
                "best_params": self.population[0].params.tolist(),
                "pop_params": [individual.params.tolist() for individual in self.population],
                "seed": self.seed,
                "config": self.config
            }
            with open(os.path.join(snapshot_dir, f"snapshot_gen_{gen}.json"), "w") as fh:
                json.dump(snapshot, fh, indent=2)

            if verbose:
                min_fitness = snapshot['min']
                avg_fitness = snapshot['avg']
                gen_time = self.history['gen_time'][-1]
                print(f"[Gen {gen}] min={min_fitness:.6g}; avg={avg_fitness:.6g}; time={gen_time:.2f}s")

        total_time = time.time() - t0

        best_individual = self.population[0]

        # --- Spaghetti Plot Simulation ---
        spaghetti_config = self.config.get("spaghetti_plot", {})
        if spaghetti_config.get("enabled", False):
            num_simulations = spaghetti_config.get("num_simulations", 100)
            noise_std_factor = spaghetti_config.get("noise_std_factor", 0.5)
            original_noise_std = self.evaluator.noise_std
            self.evaluator.noise_std *= noise_std_factor

            spaghetti_results = []
            for _ in range(num_simulations):
                _, solution = self.evaluator.simulate(best_individual.params)
                if solution:
                    spaghetti_results.append(solution.y.tolist())
            
            self.evaluator.noise_std = original_noise_std # Restaurar

        # final save
        final = {
            "history": self.history,
            "best": {
                "params": self.population[0].params.tolist(),
                "fitness": float(self.population[0].fitness)
            },
            "config": self.config,
            "total_time_s": total_time
        }
        if 'spaghetti_results' in locals():
            final["spaghetti_results"] = spaghetti_results
        with open(os.path.join(snapshot_dir, "final_result.json"), "w") as fh:
            json.dump(final, fh, indent=2)
        # close pool
        self.pool.close()
        self.pool.join()
        return final

# EOF