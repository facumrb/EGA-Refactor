"""
evaluator_toy.py
Ejemplo de un evaluador 'de juguete' (porque está simplificado) para un modelo de transcripción de 3 factores.

Este archivo simula un modelo simple de expresión génica usando Ecuaciones Diferenciales
Ordinarias (EDOs) y calcula el 'fitness' (qué tan buena es una solución) de un conjunto
de parámetros. El objetivo es encontrar los parámetros que hacen que el modelo se comporte
de la manera más parecida a un resultado objetivo o experimental.
"""

import numpy as np
from scipy.integrate import solve_ivp
import time

# Constantes para la configuración del evaluador y la función de fitness
DEFAULT_TARGET = np.array([1.0, 0.8, 0.6])

# --- Parámetros de la simulación ---
MIN_PRODUCTION_RATE = 1e-6
MIN_DEGRADATION_RATE = 1e-3
INITIAL_CONDITIONS = np.array([0.1, 0.1, 0.1])

# --- Parámetros de la función de fitness ---
FITNESS_PENALTY_FACTOR = 0.001  # Factor para la penalización por complejidad
FITNESS_FAILURE_VALUE = 1e9       # Valor de fitness si la simulación falla
REWARD_TOLERANCE = 0.1            # Tolerancia para la recompensa por alcanzar el objetivo
REACHED_REWARD_VALUE = -0.1     # Valor de la recompensa

class ToyODEEvaluator:
    """
    Clase que evalúa individuos (conjuntos de parámetros) para el modelo de EDOs.
    
    Toma un conjunto de parámetros, simula el comportamiento del sistema biológico
    descrito por las EDOs y lo compara con un resultado deseado para calcular
    un valor de 'fitness' o 'aptitud'.
    """
    def __init__(self, target=None, t_span=(0, 50), dt=0.5, noise_std=0.0):
        """Inicializa el evaluador.

        Args:
            target (np.ndarray, optional): El perfil de expresión objetivo que se quiere
            alcanzar. Si no se especifica, se usa uno por defecto. Defaults to None.
            t_span (tuple): El intervalo de tiempo para la simulación (t_inicial, t_final).
            dt (float): El paso de tiempo para la simulación.
            noise_std (float): La desviación estándar del ruido que se puede añadir a los
            datos simulados para hacer el modelo más realista. Defaults to 0.0.
        """
        # El vector de parámetros `params` que se espera tiene 9 valores:
        # 3 parámetros para cada uno de los 3 factores de transcripción (TF).
        # Para cada TF: [tasa_produccion, tasa_degradacion, escala_interaccion]
        self.t_span = t_span
        self.dt = dt
        self.noise_std = noise_std
        self.target = np.array(target, dtype=float) if target is not None else DEFAULT_TARGET

    def _ode_system(self, t, y, p):
        """Define el sistema de Ecuaciones Diferenciales Ordinarias (EDOs) que modela la red genética.

        Args:
            t (float): Tiempo actual (requerido por el solver de EDOs, aunque no se use explícitamente aquí).
            y (np.ndarray): Vector con las concentraciones actuales de las proteínas (factores de transcripción).
            p (np.ndarray): Vector de parámetros que define el comportamiento del sistema.

        Returns:
            np.ndarray: Las tasas de cambio (derivadas) de las concentraciones de proteínas.
        """
        # p (parámetros) tiene 9 elementos.
        # Modelo simple: dy_i/dt = tasa_prod_i * sigmoide(interaccion_i * suma_total) - tasa_deg_i * y_i
        # Decodificamos 'p' de forma vectorial para mayor eficiencia.
        prod = np.maximum(MIN_PRODUCTION_RATE, p[0::3]) # Tasas de producción
        deg = np.maximum(MIN_DEGRADATION_RATE, p[1::3]) # Tasas de degradación
        inter = p[2::3] # Modulan la producción según la actividad total

        # Una interacción de juguete: produccion * sigmoide(inter_i * (suma y))
        # Se calcula la suma total y la activación sigmoide de forma vectorial.
        S = np.sum(y)
        activation = 1.0 / (1.0 + np.exp(-inter * (S - 1.0)))  # Se desplaza para que la línea base importe
        
        # Se calcula la derivada de forma vectorial.
        dydt = prod * activation - deg * y
        return dydt

    def simulate(self, params):
        """Ejecuta la simulación del sistema de EDOs usando un conjunto de parámetros.

        Args:
            params (np.ndarray): El vector de parámetros a usar en la simulación.

        Returns:
            tuple: Una tupla conteniendo el estado final del sistema (y_final) y el objeto
                   de la solución completa de la simulación. Si falla, retorna (None, None).
        """
        y0 = INITIAL_CONDITIONS
        t0, tf = self.t_span
        t_eval = np.arange(t0, tf + self.dt, self.dt)
        try:
            # Se resuelve el sistema de EDOs.
            sol = solve_ivp(fun=lambda t, y: self._ode_system(t, y, params),
                            t_span=(t0, tf), y0=y0, t_eval=t_eval, vectorized=False, rtol=1e-3, atol=1e-6)
            y_final = sol.y[:, -1]
            # Opcionalmente, se añade ruido para simular variabilidad experimental.
            if self.noise_std > 0:
                y_final = y_final + np.random.normal(0, self.noise_std, size=y_final.shape)
            return y_final, sol
        except Exception as e:
            # Si la integración numérica falla, se retorna un resultado que indica el fallo.
            return None, None

    def _calculate_distance(self, y_final):
        """Calcula la distancia euclidiana entre el resultado y el objetivo."""
        return np.linalg.norm(y_final - self.target)

    def _calculate_complexity_penalty(self, params):
        """Calcula una penalización basada en la magnitud de los parámetros (regularización L1)."""
        return FITNESS_PENALTY_FACTOR * np.sum(np.abs(params))

    def _calculate_reached_reward(self, sol):
        """Otorga una recompensa si la simulación alcanza el objetivo tempranamente."""
        if sol is not None:
            for y in sol.y.T:
                if np.linalg.norm(y - self.target) < REWARD_TOLERANCE:
                    return REACHED_REWARD_VALUE  # Recompensa que reduce el fitness
        return 0.0

    def evaluate(self, params, timeout=None):
        """Evalúa un conjunto de parámetros y retorna un valor de fitness escalar.

        El fitness combina tres componentes:
        1. Distancia L2: Qué tan cerca está el resultado final del objetivo.
        2. Penalización por Complejidad: Favorece soluciones con parámetros más pequeños (parsimonia).
        3. Recompensa por Alcance: Premia a las soluciones que alcanzan el objetivo rápidamente.

        Un valor de fitness más bajo indica una mejor solución.

        Args:
            params (np.ndarray): Vector de parámetros a evaluar.
            timeout (any, optional): No implementado en esta versión. Defaults to None.

        Returns:
            float: El valor de fitness (un número más pequeño es mejor).
        """
        params = np.array(params, dtype=float)
        if len(params) < 9:
            params = np.pad(params, (0, 9 - len(params)), mode='constant', constant_values=0.1)

        y_final, sol = self.simulate(params)

        if y_final is None:
            return float(FITNESS_FAILURE_VALUE)

        # Cálculo de los componentes del fitness a través de métodos especializados
        dist = self._calculate_distance(y_final)
        complexity_penalty = self._calculate_complexity_penalty(params)
        reached_reward = self._calculate_reached_reward(sol)

        # El fitness total es la suma de sus componentes
        fitness = float(dist + complexity_penalty + reached_reward)
        return fitness

# Bloque para una prueba rápida del evaluador.
if __name__ == "__main__":
    # Se crea una instancia del evaluador.
    evalr = ToyODEEvaluator()
    # Se define un conjunto de parámetros de ejemplo.
    p = np.array([1.0, 0.1, 1.0, 0.9, 0.12, 1.1, 0.8, 0.08, 0.9])
    # Se evalúan los parámetros y se imprime el resultado de fitness.
    print("fitness:", evalr.evaluate(p))
