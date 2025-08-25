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
from typing import Dict

# Constantes para la configuración del evaluador y la función de fitness
DEFAULT_TARGET = np.array([1.0, 0.8, 0.6])
DEFAULT_BOUNDS = np.array([[0.1, 3.0], [0.01, 1.0], [-3.0, 3.0]] * 3)

# --- Parámetros de la simulación ---
MIN_PRODUCTION_RATE = 1e-6
MIN_DEGRADATION_RATE = 1e-3

# --- Parámetros de la función de fitness ---
REWARD_TOLERANCE = 0.1            # Tolerancia para la recompensa por alcanzar el objetivo
REACHED_REWARD_VALUE = -0.1     # Valor de la recompensa

class ToyODEEvaluator:
    """
    Clase que evalúa individuos (conjuntos de parámetros) para el modelo de EDOs.
    
    Toma un conjunto de parámetros, simula el comportamiento del sistema biológico
    descrito por las EDOs y lo compara con un resultado deseado para calcular
    un valor de 'fitness' o 'aptitud'.
    """
    def __init__(self, config: Dict):
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
        self.t_span = config["t_span"]
        self.dt = config["dt"]
        self.noise_std = config["noise_std"]
        self.initial_conditions = np.array(config["initial_conditions"], dtype=float)
        self.fitness_penalty_factor = config["fitness_penalty_factor"]
        self.high_fitness_penalty = config["high_fitness_penalty"]
        self.target = np.array(config["target"], dtype=float)
        self.bounds = np.array(config["bounds"], dtype=float)
        self.config = dict(config)

    def _ode_system(self, t, y, individual):
        """Define el sistema de Ecuaciones Diferenciales Ordinarias (EDOs) que modela la red genética.
        Las EDOs describen tasas de cambio (dy/dt), que biológicamente representan el balance entre síntesis (transcripción/traducción) y degradación de proteínas 
        en células. Matemáticamente, es un sistema vectorial para eficiencia; biológicamente, captura dinámicas colectivas como en circuitos genéticos reales.
        Cada EDO describe la tasa de cambio de una proteína (factor de transcripción) en función de sus interacciones con otras proteínas y su propia actividad.

        Args:
            t (float): Tiempo actual (requerido por el solver de EDOs, aunque no se use explícitamente aquí porque la tasa de cambio de las concentraciones ( dydt ) 
            depende únicamente de las concentraciones actuales ( y ) y de los parámetros del individuo ( individual_params ), pero no de t).
            y (np.ndarray): Vector con las concentraciones actuales de las proteínas (factores de transcripción).
            individual (np.ndarray): Vector de parámetros que define el comportamiento del sistema.

        Returns:
            np.ndarray: Las tasas de cambio (derivadas) de las concentraciones de proteínas.
        """
        # individual_params (parámetros) tiene 9 elementos.
        # Modelo simple: dy_i/dt = tasa_prod_i * sigmoide(interaccion_i * suma_total) - tasa_deg_i * y_i
        # Decodificamos 'individual' de forma vectorial para mayor eficiencia usando slicing ( [0::3] , etc.).
        # Slicing divide el genoma en tríos (prod, deg, inter por proteína), matemáticamente eficiente para arrays. 
        # Biológicamente, refleja cómo genes codifican tasas (ejemplo: promotores fuertes/débiles en biología molecular)
        # np.maximum previene tasas negativas, que biológicamente no ocurren (ejemplo: degradación no puede ser cero o negativa en modelos realistas).
        prod = np.maximum(MIN_PRODUCTION_RATE, individual[0::3]) # Tasas de producción
        deg = np.maximum(MIN_DEGRADATION_RATE, individual[1::3]) # Tasas de degradación
        # inter modela cómo una proteína afecta la producción de otras, simulando regulación transcripcional (ejemplo: activadores/repressores en redes genéticas).
        # Matemáticamente, es un coeficiente escalar; biológicamente, representa sensibilidad a la actividad total, como en quorum sensing donde moléculas 
        # señalan densidad celular.
        inter = individual[2::3] # Modulan la producción según la actividad total

        # Función auxiliar para sigmoide
        # Esta es una variante de la sigmoide estándar 1 / (1 + exp(-x))
        # El término -k * (x - threshold) introduce una no linealidad lineal escalada (debido al factor k que ajusta la pendiente), lo que hace que la curva sea más o menos abrupta
        # alrededor del umbral. Biológicamente, esto podría modelar sensibilidades variables en regulaciones genéticas 
        # (ejemplo: transiciones controladas en expresión génica según niveles críticos de concentración).
        def sigmoid(x, k=1.0, threshold=1.0):
            return 1.0 / (1.0 + np.exp(-k * (x - threshold)))
        # Cálculo de actividad total de la red genética
        S = np.sum(y)
        # Activación no lineal
        # El término -k * (x - threshold) se utiliza en la función sigmoide para introducir una no linealidad escalada y desplazada.
        # Matemáticamente, 'k' controla la pendiente (sensibilidad) de la transición, mientras que 'threshold' desplaza el punto de inflexión, 
        # y el signo negativo invierte la dirección para que la sigmoide crezca de 0 a 1 a medida que x supera el umbral.
        # Biológicamente, esto modela la regulación genética donde la activación ocurre solo cuando la concentración total (x) excede un umbral 
        # crítico (threshold), con 'k' representando la cooperatividad o sensibilidad de la respuesta, como en las funciones de Hill para la 
        # transcripción dependiente de factores.
        # Por ejemplo, si threshold=1.0 y k=5, para x=0.5 (por debajo del umbral), -k*(0.5-1.0)=2.5, lo que da una activación baja; 
        # para x=1.5, -k*(1.5-1.0)=-2.5, resultando en activación alta, simulando una transición abrupta en la diferenciación celular.
        activation = sigmoid(inter * S)
        # Biológicamente, el término -k * (x - threshold) en la sigmoide modela la modulación no lineal de la activación proteica, representando 
        # umbrales críticos en interacciones regulatorias, como en circuitos genéticos donde factores de transcripción activan o reprimen genes 
        # de manera cooperativa, similar a las funciones de Hill en biología de sistemas. 
        # Esto captura umbrales (threshold) y sensibilidades (k) en la expresión génica, permitiendo respuestas no lineales en procesos como la 
        # diferenciación celular o respuestas a señales.
        # En redes genéticas, x representa una señal integrada (ejemplo: concentración total de factores como en quorum sensing); 
        # -k * (x - threshold) simula la afinidad de unión ajustada por umbrales (ejemplo: cómo un factor responde solo cuando x excede threshold, 
        # con k controlando la abruptidez). Esto modela retroalimentación colectiva: proteínas "sienten" si el estado global supera un umbral y 
        # ajustan su producción, como en la diferenciación celular donde niveles críticos activan genes.

        # Derivada en forma vectorial: tasa de producción - tasa de degradación
        # tasa de producción: prod * activation
        # tasa de degradación: deg * y
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
        y0 = self.initial_conditions
        t0, tf = self.t_span
        t_eval = np.arange(t0, tf + self.dt, self.dt) # np.arange(inicio, fin, paso) son los puntos temporales
        try:
            # 2. Llamar al solucionador de EDOs (un Problema de Valor Inicial). 
            # Las EDOs son perfectas para describir cómo las concentraciones de proteínas (y) cambian con el tiempo (t), es decir, dy/dt = f(t, y).
            # El solucionador de EDOs (solve_ivp) es una función que resuelve este tipo de problemas.
            # El primer argumento (fun=lambda t, y: self._ode_system(t, y, individual)) es la función que define el sistema de EDOs:
            # le dice a solve_ivp cómo cambian las concentraciones de proteínas en cada instante.
            # El argumento fun es el más crítico para solve_ivp. Requiere una función que describa el sistema de Ecuaciones Diferenciales Ordinarias (EDOs). 
            # Esta función debe aceptar el tiempo t y el estado del sistema y (un array con las concentraciones de las especies) y devolver las derivadas ( dy/dt ), 
            # es decir, cómo están cambiando las concentraciones en ese preciso instante.
            # "lambda t, y: ..." : Aquí se usa una función anónima (lambda). Es un truco de programación para crear una función temporal y simple sin tener que definirla 
            # formalmente en otro lugar con def. solve_ivp necesita una función que solo tome t y y como argumentos principales, pero nuestro modelo (_ode_system) 
            # también necesita los parámetros del individual.
            # La función lambda actúa como un adaptador. Cuando solve_ivp la llama internamente con un tiempo t y un estado y, la lambda inmediatamente llama al método 
            # `_ode_system` (self._ode_system(t, y, individual)), pasándole no solo t y y, sino también el individual actual. El individual contiene el conjunto de parámetros 
            # (las "reglas" de la red genética) que el algoritmo genético está probando en esa evaluación específica.
            # Al principio t será t_span[0] y y será y0. Luego t será t_span[0] + dt y y será la solución de la EDO en ese t_span[0].
            # _ode_system devuelve la tasa de cambio ( dy/dt ) para cada proteína.
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
            # solve_ivp vuelve a llamar a _ode_system , pero esta vez con los nuevos valores de t y y que acaba de calcular.
            #       _ode_system calcula las nuevas derivadas para este nuevo estado.
            #       solve_ivp toma estas nuevas derivadas y calcula el estado para el siguiente pequeño paso de tiempo.
            #       Este ciclo se repite una y otra vez. solve_ivp va dando pequeños pasos en el tiempo, desde t_span[0] hasta t_span[1], 
            #       y en cada paso, llama a _ode_system para preguntarle "¿hacia dónde vamos ahora?".
            solution = solve_ivp(fun=lambda t, y: self._ode_system(t, y, individual), t_span=(t0, tf), y0=y0,
                                    t_eval=t_eval, vectorized=False, rtol=1e-3, atol=1e-6, method="LSODA", 
                                    dense_output=True)
            if solution.status != 0:
                print(f"Error en la simulación (por solution): {solution.message}")
                return None, None
            # 3. Extraer el estado final del sistema.
            # solution.y es la matriz de resultados. [:, -1] es una forma de seleccionar
            # de todas las filas (:) la última columna (-1). 
            # Esto nos da un array con la concentración de cada proteína en el tiempo final 'tf'.
            y_final = solution.y[:, -1]
            # Opcionalmente, se añade ruido para simular variabilidad experimental.
            if self.noise_std > 0:
                np.random.seed(self.seed) # Semilla fija para reproducibilidad del ruido
                y_final = y_final + np.random.normal(0, self.noise_std, size=y_final.shape)
            return y_final, solution
        except Exception:
            # Si la integración numérica falla, se retorna un resultado que indica el fallo.
            return None, None

    def _calculate_L2_distance(self, y_final):
        """Calcula la distancia euclidiana (linea recta L2) entre el resultado y el objetivo."""
        # np.linalg.norm() calcula la norma (distancia) entre dos vectores.
        # Representa cuán "lejos" están los dos puntos en el espacio.
        # Biológicamente, esta distancia representa cuán cerca está el estado final de la red genética modelada del 
        # estado deseado en términos de niveles de expresión génica o concentraciones de proteínas.
        return np.linalg.norm(y_final - self.target) # Siempre es positivo

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
        # El signo del número que retorna (+/-) es el mismo signo del fitness_penalty_factor
        return self.fitness_penalty_factor * np.sum(np.abs(individual))

    def _calculate_reward_tolerance(self):
        """Calcula REWARD_TOLERANCE dinámicamente basado en la norma del target y factores de configuración.

        Biológicamente, simula umbrales de activación en redes genéticas, adaptándose a la escala de expresión
        para capturar variabilidad celular real (ej. ruido en transcripción).
        """
        base_tolerance = 0.1  # Valor base empírico (10% de desviación aceptable)
        scale_factor = np.linalg.norm(self.target)  # Escala con magnitud del objetivo
        noise_factor = 1 + self.noise_std  # Ajuste por variabilidad biológica/numérica
        return base_tolerance * scale_factor * noise_factor

    def _calculate_reached_reward(self, solution):
        """Otorga una recompensa si la simulación alcanza el objetivo tempranamente.

        Biológicamente, esta recompensa modela la ventaja adaptativa de redes genéticas que logran
        un estado deseado (ej. patrón de expresión en diferenciación celular) de manera rápida y
        eficiente, simulando presiones evolutivas por respuestas oportunas en entornos dinámicos.
        """
        if solution is not None:
            for system_state, y in enumerate(solution.y.T):
                # solution.y es una matriz NumPy donde cada fila representa la evolución temporal de una proteína (concentración a lo largo del tiempo), 
                # y cada columna un instante temporal. ".T" la transpone, convirtiéndola en una matriz donde cada fila es un vector de concentraciones 
                # de todas las proteínas en un tiempo específico. Así, solution.y.T contiene los estados del sistema (vectores de concentraciones) para 
                # cada punto de tiempo evaluado en la simulación.
                if self._calculate_L2_distance(y) < self._calculate_reward_tolerance():
                    # un estado biológico "deseado" no necesita ser exacto debido a la variabilidad inherente en los sistemas biológicos, como 
                    # fluctuaciones en la expresión génica o ruido estocástico. Permite que el modelo considere como "alcanzado" un estado cercano 
                    # al objetivo, reflejando tolerancias reales en procesos como la regulación génica, donde leves desviaciones son aceptables sin 
                    # comprometer la funcionalidad celular.
                    t_reached = solution.t[system_state]
                    # Se asigna el valor del tiempo de simulación en el índice system_state , que representa el primer punto temporal donde se cumple 
                    # la condición de proximidad al objetivo.

                    # Se calcula una recompensa negativa que es inversamente proporcional al tiempo t_reached (cuanto menor el tiempo, mayor la magnitud 
                    # negativa de la recompensa, lo que mejora el fitness ya que valores más bajos de fitness son mejores). Se añade 1e-6 para evitar 
                    # división por cero si t_reached es cero. Esto penaliza tiempos largos y premia respuestas rápidas.
                    # Modela la ventaja evolutiva de redes genéticas que logran estados deseados rápidamente, como en respuestas inmunes (e.g., activación 
                    # rápida de genes ante patógenos) o desarrollo embrionario (e.g., diferenciación celular oportuna). Una recompensa más negativa 
                    # incentiva eficiencia temporal, reflejando selección natural por mecanismos biológicos ágiles y energéticamente óptimos.
                    return -REACHED_REWARD_VALUE / (t_reached + 1e-6)
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
            timeout (any, optional): No implementado en esta versión. Defaults to None.

        Returns:
            float: El valor de fitness (un número más pequeño es mejor).
        """
        individual = np.array(individual, dtype=float)
        # Convierte la lista de parámetros en un array de NumPy de tipo flotante, el formato que necesita 
        # el solver de Ecuaciones Diferenciales (ODE).

        # validaciones básicas
        if self.bounds is None or individual.shape[0] != self.bounds.shape[0]:
            print("Error: bounds no están definidos o no coinciden con la longitud del individuo.")
            return (float(self.high_fitness_penalty), None)

        # Soft constraint para bounds en lugar de 'inf'
        bound_violation = np.sum(np.maximum(0, self.bounds[:, 0] - individual) + np.maximum(0, individual - self.bounds[:, 1]))
        if bound_violation > 0:
            penalty = bound_violation * 1000  # Penalización proporcional, ajustable
        else:
            penalty = 0

        try:
            # Estado final del sistema y el objeto de la solución
            y_final, solution = self.simulate(individual)
            if y_final is None or solution is None:
                return float(self.high_fitness_penalty), None

            if y_final is None:
                print("Error: y_final es None.")
                return float(self.high_fitness_penalty + penalty), None  # Penalización alta pero finita

            # Cálculo de los componentes del fitness a través de métodos especializados
            L2_distance = self._calculate_L2_distance(y_final)
            complexity_penalty = self._calculate_complexity_penalty(individual)
            reached_reward = self._calculate_reached_reward(solution)

            # El fitness total es la suma de sus componentes
            fitness = float(L2_distance + complexity_penalty + reached_reward)
            return fitness, solution
        except Exception as error:
            # loguear error para debugging
            print(f"[Evaluator] excepción al evaluar: {error}")
            return float(self.high_fitness_penalty), None

# Bloque para una prueba rápida del evaluador.
if __name__ == "__main__":
    # Se crea una instancia del evaluador.
    evaluator = ToyODEEvaluator()
    # Se define un conjunto de parámetros de ejemplo.
    p = np.array([1.0, 0.1, 1.0, 0.9, 0.12, 1.1, 0.8, 0.08, 0.9])
    # Se evalúan los parámetros y se imprime el resultado de fitness.
    print("fitness:", evaluator.evaluate(p))
