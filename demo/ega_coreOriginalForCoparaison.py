"""
ega_core.py
Núcleo de Algoritmo Genético Elitista (EGA) adaptado para vectores reales y evaluadores pluggable.
Características:
- Individual como vector real con bounds
- Selección torneo (por defecto) y opción ruleta (corregida)
- Crossover BLX-alpha (reales)
- Mutación gaussiana adaptativa
- Elitismo (preserva top-K)
- Evaluación con multiprocessing pool y timeout
- Caching de evaluaciones
- Snapshots JSON por generación
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

# -----------------------
# Utilidades / configuración
# -----------------------
def init_worker(seed_base: int):
    """Initializer para workers: fija semilla por worker para reproducibilidad."""
    import os, random, numpy as _np
    pid = os.getpid()
    s = (seed_base + pid) % (2**31 - 1)
    random.seed(s)
    _np.random.seed(s)

def safe_round_tuple(arr, digits=6):
    return tuple([round(float(x), digits) for x in arr])

# -----------------------
# Clase Individual
# -----------------------
class Individual:
    def __init__(self, bounds: np.ndarray, init_strategy: str = "uniform"):
        """
        bounds: array shape (n_params, 2) with min, max
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
        """Decoder: returns params array (could apply transforms/repairs here)."""
        # Repair to bounds (safety)
        self.params = np.clip(self.params, self.bounds[:,0], self.bounds[:,1])
        return self.params

    def copy(self):
        ind = Individual(self.bounds)
        ind.params = np.array(self.params, copy=True)
        ind.fitness = self.fitness
        return ind

# -----------------------
# Operadores
# -----------------------
def blx_alpha_crossover(p1: np.ndarray, p2: np.ndarray, alpha=0.3, bounds=None):
    n = len(p1)
    child = np.zeros(n, dtype=float)
    for i in range(n):
        cmin = min(p1[i], p2[i])
        cmax = max(p1[i], p2[i])
        I = cmax - cmin
        low = cmin - alpha * I
        high = cmax + alpha * I
        # Respect global bounds if provided
        if bounds is not None:
            low = max(low, bounds[i,0])
            high = min(high, bounds[i,1])
        child[i] = random.uniform(low, high)
    if bounds is not None:
        child = np.clip(child, bounds[:,0], bounds[:,1])
    return child

def gaussian_mutation(params: np.ndarray, mutation_rate: float, mutation_scale: np.ndarray, bounds=None):
    """
    mutation_scale: per-parameter std fraction of (max-min), e.g. 0.05 for 5%
    """
    n = len(params)
    for i in range(n):
        if random.random() < mutation_rate:
            std = mutation_scale[i]
            # absolute std
            abs_std = std * (bounds[i,1] - bounds[i,0]) if bounds is not None else std
            params[i] += np.random.normal(0, abs_std)
    if bounds is not None:
        params = np.clip(params, bounds[:,0], bounds[:,1])
    return params

# -----------------------
# EGA core
# -----------------------
class EGA:
    def __init__(self, config: Dict, evaluator):
        """
        config: dict with keys (populationSize, generations, crossover_rate, mutation_rate,
                                 elite_size, bounds, seed, processes, alpha_blx, mutation_scale, timeout)
        evaluator: object with method evaluate(params: np.ndarray) -> float (lower better)
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
        self.cache = {}  # caching evaluations: key -> fitness
        self.population = [Individual(self.bounds) for _ in range(self.pop_size)]
        self.history = {"min": [], "avg": [], "gen_time": []}
        # Pool
        self.pool = Pool(processes=self.processes, initializer=init_worker, initargs=(self.seed,))

    # ----- fitness evaluation wrappers -----
    def _eval_single(self, individual: Individual):
        key = safe_round_tuple(individual.decode())
        if key in self.cache:
            return self.cache[key]
        # apply evaluator with timeout via pool
        try:
            # apply async and get with timeout
            async_res = self.pool.apply_async(self.evaluator.evaluate, (individual.params,))
            fitness = async_res.get(timeout=self.timeout)
        except Exception as e:
            fitness = float(1e9)  # large penalty
        # store in cache
        self.cache[key] = float(fitness)
        return float(fitness)

    def evaluate_population(self, population=None):
        if population is None:
            population = self.population
        # Evaluate in parallel using pool.map-like approach with wrapper
        args = population
        # For each individual evaluate
        results = []
        for ind in args:
            results.append(self._eval_single(ind))
        # assign
        for ind, f in zip(population, results):
            ind.fitness = f
        return results

    # ----- selection -----
    def tournament_selection(self, k=3, num_select=None):
        if num_select is None:
            num_select = self.pop_size
        selected = []
        for _ in range(num_select):
            participants = random.sample(self.population, k)
            winner = min(participants, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def roulette_selection(self, num_select=None):
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

    # ----- main GA loop -----
    def run(self, snapshot_dir="snapshots", verbose=True):
        os.makedirs(snapshot_dir, exist_ok=True)
        # initial evaluation
        t0 = time.time()
        self.evaluate_population(self.population)
        gen = 0
        self._record_stats(gen)
        if verbose:
            print(f"[Gen {gen}] min={min([p.fitness for p in self.population]):.6g}; avg={np.mean([p.fitness for p in self.population]):.6g}")

        for gen in range(1, self.generations+1):
            start = time.time()
            # Elitism: preserve top-K
            sorted_pop = sorted(self.population, key=lambda x: x.fitness)
            elites = [ind.copy() for ind in sorted_pop[:self.elite_size]]

            # Selection
            parents = self.tournament_selection(k=int(self.config.get("tournament_k", 3)), num_select=self.pop_size)

            # Create offspring
            offspring = []
            for i in range(0, self.pop_size//2):
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                # crossover?
                if random.random() < self.crossover_rate:
                    child1_params = blx_alpha_crossover(p1.params, p2.params, alpha=self.alpha_blx, bounds=self.bounds)
                    child2_params = blx_alpha_crossover(p2.params, p1.params, alpha=self.alpha_blx, bounds=self.bounds)
                else:
                    child1_params = np.array(p1.params, copy=True)
                    child2_params = np.array(p2.params, copy=True)
                # mutation
                child1_params = gaussian_mutation(child1_params, self.mutation_rate, self.mutation_scale, bounds=self.bounds)
                child2_params = gaussian_mutation(child2_params, self.mutation_rate, self.mutation_scale, bounds=self.bounds)

                # build Individual
                c1 = Individual(self.bounds); c1.params = child1_params
                c2 = Individual(self.bounds); c2.params = child2_params
                offspring.extend([c1, c2])

            # If odd size, trim or add one
            if len(offspring) > self.pop_size:
                offspring = offspring[:self.pop_size]

            # Evaluate offspring
            self.evaluate_population(offspring)

            # Form new population: elites + best of offspring until fill
            combined = elites + offspring
            combined_sorted = sorted(combined, key=lambda x: x.fitness)
            self.population = [ind.copy() for ind in combined_sorted[:self.pop_size]]

            # record stats
            self._record_stats(gen)
            end = time.time()
            self.history["gen_time"].append(end - start)

            # snapshot
            snapshot = {
                "gen": gen,
                "min": float(min([p.fitness for p in self.population])),
                "avg": float(np.mean([p.fitness for p in self.population])),
                "best_params": self.population[0].params.tolist(),
                "seed": self.seed,
                "config": self.config
            }
            with open(os.path.join(snapshot_dir, f"snapshot_gen_{gen}.json"), "w") as fh:
                json.dump(snapshot, fh, indent=2)

            if verbose:
                print(f"[Gen {gen}] min={snapshot['min']:.6g}; avg={snapshot['avg']:.6g}; time={self.history['gen_time'][-1]:.2f}s")

        total_time = time.time() - t0
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
        with open(os.path.join(snapshot_dir, "final_result.json"), "w") as fh:
            json.dump(final, fh, indent=2)
        # close pool
        self.pool.close()
        self.pool.join()
        return final

    def _record_stats(self, gen):
        fits = np.array([ind.fitness for ind in self.population], dtype=float)
        self.history["min"].append(float(np.min(fits)))
        self.history["avg"].append(float(np.mean(fits)))

# EOF
