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
DEFAULT_BOUNDS = np.array([[0.1, 3.0], [0.01, 1.0], [-3.0, 3.0]] * 3)

# --- Parámetros de la simulación ---
MIN_PRODUCTION_RATE = 1e-6
MIN_DEGRADATION_RATE = 1e-3
INITIAL_CONDITIONS = np.array([0.1, 0.1, 0.1])
# Estado Basal de Expresión : El valor 0.1 para cada uno de los tres componentes 
# del vector representa una concentración inicial, baja pero no nula, de los tres 
# factores de transcripción (o proteínas) en el modelo. Biológicamente, es muy 
# raro que la concentración de una proteína sea absolutamente cero. Las células 
# suelen tener un nivel de "fugas" o expresión basal de muchos genes.
# Punto de Partida para la Reprogramación : En el contexto de la reprogramación 
# celular (que es lo que este modelo simula de forma abstracta), estas condiciones 
# iniciales representan el estado de una célula antes de que se aplique cualquier 
# estímulo. El objetivo del algoritmo genético es encontrar los parámetros 
# (las "reglas" de interacción genética) que, partiendo de este estado basal, lleven 
# a la célula a un estado final deseado (el target ).
# Evitar Singularidades Matemáticas : Desde un punto de vista computacional, empezar 
# con concentraciones exactamente en cero puede, en algunos modelos, llevar a resultados 
# triviales (si no hay nada, nada cambia) o a inestabilidades numéricas en el solver 
# de las ecuaciones diferenciales. Empezar con un valor pequeño y positivo asegura que 
# el sistema pueda "arrancar" y evolucionar.
# En resumen, [0.1, 0.1, 0.1] es una suposición plausible para un estado celular no 
# estimulado, donde los factores de interés están presentes en pequeñas cantidades, 
# pero están listos para responder a las interacciones que el algoritmo genético descubrirá.


# --- Parámetros de la función de fitness ---
FITNESS_PENALTY_FACTOR = 0.001  # Factor para la penalización por complejidad
# Penalización por complejidad : Esta constante, FITNESS_PENALTY_FACTOR, se utiliza para multiplicar 
# la complejidad del individuo (en este caso, la suma de sus parámetros) y agregar una penalización 
# al fitness. Esto incentiva a los algoritmos genéticos a encontrar soluciones más simples (menos 
# complejas) cuando es posible.
# La mejor aproximación es experimentar. Ejecutar el algoritmo con diferentes valores del factor de 
# penalización y analizar los resultados.

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
    def __init__(self, target=None, bounds=None, t_span=(0, 50), dt=0.5, noise_std=0.0):
        """Inicializa el evaluador.

        Args:
            target (np.ndarray, optional): El perfil de expresión objetivo que se quiere
            alcanzar. Si no se especifica, se usa uno por defecto. Defaults to None.
            bounds (np.ndarray, optional): Los límites para cada parámetro. Hace a la plausibilidad
            de los parámetros más razonable. Si no se especifica, se usa uno por defecto. Defaults to None.
            t_span (tuple): El intervalo de tiempo para la simulación (t_inicial, t_final).
            dt (float): El paso de tiempo para la simulación.
            noise_std (float): La desviación estándar del ruido que se puede añadir a los
            datos simulados para hacer el modelo más realista. Defaults to 0.0.
        """
        self.t_span = t_span
        self.dt = dt
        self.noise_std = noise_std
        self.target = np.array(target, dtype=float) if target is not None else DEFAULT_TARGET
        self.bounds = np.array(bounds, dtype=float) if bounds is not None else DEFAULT_BOUNDS


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

    def simulate(self, individual):
        """Ejecuta la simulación del sistema de EDOs usando un conjunto de parámetros.

        Args:
            params (np.ndarray): El vector de parámetros a usar en la simulación.

        Returns:
            tuple: Una tupla conteniendo el estado final del sistema (y_final) y el objeto
                   de la solución completa de la simulación. Si falla, retorna (None, None).
        """
        y0 = INITIAL_CONDITIONS
        t0, tf = self.t_span
        t_eval = np.arange(t0, tf + self.dt, self.dt) # np.arange(inicio, fin, paso) son los puntos temporales
        try:
            # Se resuelve el sistema de EDOs.
            solution = solve_ivp(fun=lambda t, y: self._ode_system(t, y, individual),
                            t_span=(t0, tf), y0=y0, t_eval=t_eval, vectorized=False, rtol=1e-3, atol=1e-6)
            y_final = solution.y[:, -1]
            # Opcionalmente, se añade ruido para simular variabilidad experimental.
            if self.noise_std > 0:
                y_final = y_final + np.random.normal(0, self.noise_std, size=y_final.shape)
            return y_final, solution
        except Exception as e:
            # Si la integración numérica falla, se retorna un resultado que indica el fallo.
            return None, None

    def _calculate_L2_distance(self, y_final):
        """Calcula la distancia euclidiana (linea recta L2) entre el resultado y el objetivo."""
        return np.linalg.norm(y_final - self.target)
        # np.linalg.norm() calcula la norma (distancia) entre dos vectores.

    def _calculate_complexity_penalty(self, individual):
        """
        Calcula una penalización basada en la magnitud (valor absoluto) de los parámetros (regularización L1).
        No nos interesa si una interacción es inhibidora (negativa) o activadora (positiva), solo su magnitud o "fuerza".
        Se suma todos los valores absolutos calculados en el paso anterior. El resultado es un único número que 
        representa la "magnitud total" de todos los parámetros del individuo.
        Finalmente, multiplica esa suma por la constante FITNESS_PENALTY_FACTOR. 
        FITNESS_PENALTY_FACTOR actúa como un peso: decide cuánta importancia se le da a esta penalización de complejidad 
        en el cálculo total del fitness.
        """
        return FITNESS_PENALTY_FACTOR * np.sum(np.abs(individual))

    def _calculate_reached_reward(self, solution):
        """Otorga una recompensa si la simulación alcanza el objetivo tempranamente."""
        if solution is not None:
            for y in solution.y.T:
                if np.linalg.norm(y - self.target) < REWARD_TOLERANCE:
                    return REACHED_REWARD_VALUE  # Recompensa que reduce el fitness
        return 0.0

    def evaluate(self, individual, timeout=None):
        """Evalúa un individuo y retorna un valor de fitness escalar.

        El fitness combina tres componentes:
        1. Distancia L2: Qué tan cerca está el resultado final del objetivo.
        2. Penalización por Complejidad: Favorece soluciones con parámetros más pequeños (parsimonia).
        3. Recompensa por Alcance: Premia a las soluciones que alcanzan el objetivo rápidamente.

        Un valor de fitness más bajo indica una mejor solución.

        Args:
            individual (np.ndarray): Vector de parámetros a evaluar.
            timeout (any, optional): No implementado en esta versión. Defaults to None.

        Returns:
            float: El valor de fitness (un número más pequeño es mejor).
        """
        individual = np.array(individual, dtype=float)
        # Convierte la lista de parámetros en un array de NumPy de tipo flotante, el formato que necesita 
        # el solver de Ecuaciones Diferenciales (ODE).
        """
        No es recomendable rellenar con valores por defecto, ya que los parámetros que no se hayan
        optimizado pueden tener un impacto negativo en el rendimiento del modelo:

        num_params = self.bounds.shape[0]
        if len(individual) < num_params:
            individual = np.pad(individual, (0, num_params - len(individual)), mode='constant', constant_values=0.1)
        # Se verifica si los parámetros están dentro de los límites.
        if not np.all((self.bounds[:, 0] <= individual) & (individual <= self.bounds[:, 1])):
            return float(FITNESS_FAILURE_VALUE)
        
        Este código es un "parche" para evitar un fallo del EGA, añadiendo "genes" por defecto.
        """
        y_final, solution = self.simulate(individual)

        if y_final is None:
            return float(FITNESS_FAILURE_VALUE)

        # Cálculo de los componentes del fitness a través de métodos especializados
        L2_distance = self._calculate_L2_distance(y_final)
        # VER RECOMPENZA PROPORCIONAL AL TIEMPO
        # Hacer la recompensa inversamente proporcional al tiempo de la simulación en que se alcanza. 
        # Es decir, cuanto antes se llegue, mayor es la recompensa (menor el valor de fitness).
        complexity_penalty = self._calculate_complexity_penalty(individual)
        reached_reward = self._calculate_reached_reward(solution)

        # El fitness total es la suma de sus componentes
        fitness = float(L2_distance + complexity_penalty + reached_reward)
        return fitness

# Bloque para una prueba rápida del evaluador.
if __name__ == "__main__":
    # Se crea una instancia del evaluador.
    evaluator = ToyODEEvaluator()
    # Se define un conjunto de parámetros de ejemplo.
    p = np.array([1.0, 0.1, 1.0, 0.9, 0.12, 1.1, 0.8, 0.08, 0.9])
    # Se evalúan los parámetros y se imprime el resultado de fitness.
    print("fitness:", evaluator.evaluate(p))
