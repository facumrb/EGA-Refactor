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


# --- Parámetros de la función de fitness ---

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
    def __init__(self, target=None, bounds=None, t_span=(0, 50), dt=0.5, noise_std=0.0, fitness_penalty_factor=0.001, initial_conditions=None):

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
            fitness_penalty_factor (float): Factor para penalizar la complejidad en la función de fitness.
        """
        self.t_span = t_span
        self.dt = dt
        self.noise_std = noise_std
        self.fitness_penalty_factor = fitness_penalty_factor
        self.initial_conditions = np.array(initial_conditions, dtype=float)
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
        t0, tf = self.t_span
        t_eval = np.arange(t0, tf + self.dt, self.dt) # np.arange(inicio, fin, paso) son los puntos temporales
        # 1. Iniciar un bloque de manejo de errores.
        # La simulación numérica a veces puede fallar (por ejemplo, si los parámetros del individuo
        # crean un sistema inestable), y este 'try' nos permite capturar esos fallos sin que el programa se detenga.
        try:
            # 2. Llamar al solucionador de EDOs (un Problema de Valor Inicial). 
            # Las EDOs son perfectas para describir cómo las concentraciones de proteínas (y) cambian con el tiempo (t), es decir, dy/dt = f(t, y).
            # La función lambda le dice a solve_ivp cómo cambian las concentraciones de proteínas en cada instante.
            # "lambda t, y: self._ode_system(t, y, individual)" es una forma corta de definir una función. 
            # Llama a self._ode_system, pasándole el tiempo actual t, las concentracions actuales y, y los parámetros del individuo ( individual ) 
            # que el algoritmo genético está probando. 
            # _ode_system devuelve la tasa de cambio ( dy/dt ) para cada proteína.
            solution = solve_ivp(fun=lambda t, y: self._ode_system(t, y, individual),
                            t_span=self.t_span, y0=self.initial_conditions, t_eval=t_eval, vectorized=False, rtol=1e-3, atol=1e-6)
            # t_span es el intervalo de tiempo (t0, tf) en el que se integra.
            # y0 son las condiciones iniciales (concentraciones iniciales de proteínas).
            # t_eval es la lista de tiempos en los que se desea obtener la solución.
            # vectorized=False indica que la función _ode_system no está vectorizada.
            # rtol es la tolerancia relativa para la integración.
            # atol es la tolerancia absoluta para la integración.
            # solve_ivp no devuelve un objeto que contiene toda la información sobre la simulación.
            # Los atributos más importantes de este objeto solution son:
            # solution.t: Un array con los tiempos en los que se evaluó la solución. Coincide con t_eval.
            # solution.y: El resultado principal: es una matriz con las concentraciones de proteínas en cada tiempo.
            # Cada fila de solution.y corresponde a una proteína, y cada columna a un tiempo.
            # solution.y.shape es (3, 101), donde 3 es el número de proteínas y 101 es el número de tiempos.
            
            # 3. Extraer el estado final del sistema.
            # solution.y es la matriz de resultados. [:, -1] es una forma de seleccionar
            # de todas las filas (:) la última columna (-1). 
            # Esto nos da un array con la concentración de cada proteína en el tiempo final 'tf'.
            y_final = solution.y[:, -1]
            # 4. (Opcional) Simular ruido experimental.
            # Los experimentos biológicos reales no son perfectos. Esta línea añade un poco de
            # aleatoriedad (ruido gaussiano) al resultado final para que la simulación sea más realista.
            if self.noise_std > 0:
                y_final = y_final + np.random.normal(0, self.noise_std, size=y_final.shape)
            # 5. Devolver el resultado exitoso.
            # Se devuelve tanto el estado final como el objeto 'solution' completo, por si se necesita más adelante.
            return y_final, solution
        # 6. Capturar cualquier error que haya ocurrido durante el 'try'.
        except Exception:
            # 7. Si la simulación falló, se informa y se devuelve un resultado que indica el fallo ('None').
            # Esto es crucial para que el algoritmo genético sepa que el individuo que causó el error
            # no es una solución viable y le asigne un fitness muy malo.
            return None, None

    def _calculate_L2_distance(self, y_final):
        """Calcula la distancia euclidiana (linea recta L2) entre el resultado y el objetivo."""
        return np.linalg.norm(y_final - self.target) # Siempre es positivo
        # np.linalg.norm() calcula la norma (distancia) entre dos vectores.

    def _calculate_complexity_penalty(self, individual):
        """
        Calcula una penalización basada en la magnitud (valor absoluto) de los parámetros (regularización L1).
        No nos interesa si una interacción es inhibidora (negativa) o activadora (positiva), solo su magnitud o "fuerza".
        Se suma todos los valores absolutos calculados en el paso anterior. El resultado es un único número que 
        representa la "magnitud total" de todos los parámetros del individuo.
        Finalmente, multiplica esa suma por la constante fitness_penalty_factor. 
        fitness_penalty_factor actúa como un peso: decide cuánta importancia se le da a esta penalización de complejidad 
        en el cálculo total del fitness.
        """
        return self.fitness_penalty_factor * np.sum(np.abs(individual))

        # El signo del número que retorna (+/-) es el mismo signo del fitness_penalty_factor

    def _calculate_reached_reward(self, solution):
        """Otorga una recompensa si la simulación alcanza el objetivo tempranamente."""
        if solution is not None:
            for y in solution.y.T:
                if np.linalg.norm(y - self.target) < REWARD_TOLERANCE:
                    return REACHED_REWARD_VALUE  # Recompensa que reduce el fitness
        return 0.0

    def evaluate(self, individual):
        """Evalúa un individuo y retorna un valor de fitness escalar.

        El fitness combina tres componentes:
        1. Distancia L2: Qué tan cerca está el resultado final del objetivo.
        2. Penalización por Complejidad: Favorece soluciones con parámetros más pequeños (parsimonia).
        3. Recompensa por Alcance: Premia a las soluciones que alcanzan el objetivo rápidamente.

        Un valor de fitness más bajo indica una mejor solución.

        Args:
            individual (np.ndarray): Vector de parámetros a evaluar.

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
        # Estado final del sistema y el objeto de la solución

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
