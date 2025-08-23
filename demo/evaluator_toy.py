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
    def __init__(self, config: dict):

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
        self.t_span = config["t_span"]
        self.dt = config["dt"]
        self.noise_std = config["noise_std"]
        self.fitness_penalty_factor = config["fitness_penalty_factor"]
        self.initial_conditions = np.array(config["initial_conditions"], dtype=float)
        self.min_production_rate = config["min_production_rate"]
        self.min_degradation_rate = config["min_degradation_rate"]
        self.target = np.array(config["target"], dtype=float) if config["target"] is not None else DEFAULT_TARGET
        self.bounds = np.array(config["bounds"], dtype=float) if config["bounds"] is not None else DEFAULT_BOUNDS

    def _ode_system(self, t, y, individual_params):
        """Define el sistema de Ecuaciones Diferenciales Ordinarias (EDOs) que modela la red genética.
        Las EDOs describen tasas de cambio (dy/dt), que biológicamente representan el balance entre síntesis (transcripción/traducción) y degradación de proteínas 
        en células. Matemáticamente, es un sistema vectorial para eficiencia; biológicamente, captura dinámicas colectivas como en circuitos genéticos reales.
        Cada EDO describe la tasa de cambio de una proteína (factor de transcripción) en función de sus interacciones con otras proteínas y su propia actividad.

        Args:
            t (float): Tiempo actual (requerido por el solver de EDOs, aunque no se use explícitamente aquí porque la tasa de cambio de las concentraciones ( dydt ) 
            depende únicamente de las concentraciones actuales ( y ) y de los parámetros del individuo ( individual_params ), pero no de t).
            y (np.ndarray): Vector con las concentraciones actuales de las proteínas (factores de transcripción).
            individual_params (np.ndarray): Vector de parámetros que define el comportamiento del sistema.

        Returns:
            np.ndarray: Las tasas de cambio (derivadas) de las concentraciones de proteínas.
        """
        # individual_params (parámetros) tiene 9 elementos.
        # Modelo simple: dy_i/dt = tasa_prod_i * sigmoide(interaccion_i * suma_total) - tasa_deg_i * y_i
        # Decodificamos 'individual_params' de forma vectorial para mayor eficiencia usando slicing ( [0::3] , etc.).
        # Slicing divide el genoma en tríos (prod, deg, inter por proteína), matemáticamente eficiente para arrays. 
        # Biológicamente, refleja cómo genes codifican tasas (ejemplo: promotores fuertes/débiles en biología molecular)
        # np.maximum previene tasas negativas, que biológicamente no ocurren (ejemplo: degradación no puede ser cero o negativa en modelos realistas).
        prod = np.maximum(self.min_production_rate, individual_params[0::3]) # Tasas de producción de proteínas (transcripción génica)
        deg = np.maximum(self.min_degradation_rate, individual_params[1::3]) # Tasas de degradación (turnover proteico)
        # inter modela cómo una proteína afecta la producción de otras, simulando regulación transcripcional (ejemplo: activadores/repressores en redes genéticas).
        # Matemáticamente, es un coeficiente escalar; biológicamente, representa sensibilidad a la actividad total, como en quorum sensing donde moléculas 
        # señalan densidad celular.
        inter = individual_params[2::3] # Modulan la producción según la actividad total

        # Función auxiliar para sigmoide
        # Esta es una variante de la sigmoide estándar 1 / (1 + exp(-x))
        # El término -k * (x - threshold) introduce una no linealidad lineal escalada (debido al factor k que ajusta la pendiente), lo que hace que la curva sea más o menos abrupta
        # alrededor del umbral. Biológicamente, esto podría modelar sensibilidades variables en regulaciones genéticas 
        # (ejemplo: transiciones controladas en expresión génica según niveles críticos de concentración).
        def sigmoid(x, k=1.0, threshold=1.0):
            return 1.0 / (1.0 + np.exp(-x * (-k - threshold)))
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
        t0, tf = self.t_span
        t_eval = np.arange(t0, tf + self.dt, self.dt) # np.arange(inicio, fin, paso) son los puntos temporales
        # 1. Iniciar un bloque de manejo de errores.
        # La simulación numérica a veces puede fallar (por ejemplo, si los parámetros del individuo
        # crean un sistema inestable), y este 'try' nos permite capturar esos fallos sin que el programa se detenga.
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
            solution = solve_ivp(fun=lambda t, y: self._ode_system(t, y, individual),
                            t_span=self.t_span, y0=self.initial_conditions, t_eval=t_eval, vectorized=False, rtol=1e-3, atol=1e-6)
            
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
        except Exception as error:
            # 7. Si la simulación falló, se informa y se devuelve un resultado que indica el fallo ('None').
            # Esto es crucial para que el algoritmo genético sepa que el individuo que causó el error
            # no es una solución viable y le asigne un fitness muy malo.
            raise ValueError(f"Simulación fallida para individuo: {error}")

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

        # Se verifica si los parámetros están dentro de los límites.
        # Si no están, se penaliza con un valor infinito.
        if not np.all((self.bounds[:, 0] <= individual) & (individual <= self.bounds[:, 1])):
            return float('inf') # Penalización infinita para bounds violados
        
        try:
            # Estado final del sistema y el objeto de la solución
            y_final, solution = self.simulate(individual)
            if y_final is None:  # Si aún hay fallo (aunque ahora usamos excepciones)
                return float('inf')

            # Cálculo de los componentes del fitness a través de métodos especializados
            L2_distance = self._calculate_L2_distance(y_final)
            complexity_penalty = self._calculate_complexity_penalty(individual)
            reached_reward = self._calculate_reached_reward(solution)
            # VER RECOMPENZA PROPORCIONAL AL TIEMPO
            # Hacer la recompensa inversamente proporcional al tiempo de la simulación en que se alcanza. 
            # Es decir, cuanto antes se llegue, mayor es la recompensa (menor el valor de fitness).

            # El fitness total es la suma de sus componentes
            fitness = float(L2_distance + complexity_penalty + reached_reward)
            return fitness
        except ValueError:
            return float('inf') # Penalización para fallos en simulación

# Bloque para una prueba rápida del evaluador.
if __name__ == "__main__":
    # Se crea una instancia del evaluador.
    evaluator = ToyODEEvaluator()
    # Se define un conjunto de parámetros de ejemplo.
    p = np.array([1.0, 0.1, 1.0, 0.9, 0.12, 1.1, 0.8, 0.08, 0.9])
    # Se evalúan los parámetros y se imprime el resultado de fitness.
    print("fitness:", evaluator.evaluate(p))
