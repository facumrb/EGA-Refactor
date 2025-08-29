import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al snapshot final (se espera que incluya la población final)
snapshot_dir = "snapshots"
final_result_path = os.path.join(snapshot_dir, "final_result.json")

# Cargar los datos del snapshot final
with open(final_result_path, "r") as f:
    data = json.load(f)

# Se asume que final_result.json contiene una lista de individuos en la clave "population"
# Cada individuo debe tener, al menos, las llaves "params" (lista de valores numéricos)
# y "fitness" (valor numérico)
population = data.get("population", [])
if not population:
    raise ValueError("No se encontró la población en final_result.json. Asegúrate de que el snapshot incluye la información de la población final.")

fitness = []
complexity = []

# Para cada individuo, calculamos la complejidad como la suma de los valores absolutos de sus parámetros
for individual in population:
    params = individual.get("params")
    fit = individual.get("fitness")
    if params is not None and fit is not None:
        fitness.append(fit)
        comp = sum(abs(x) for x in params)
        complexity.append(comp)

""" REVISAR (es probable que haya errores según cómo se define la interpretación del fitness):
La "complejidad", definida como la suma de los valores absolutos de los parámetros, puede interpretarse biológicamente 
como una medida global de la intensidad o magnitud de las interacciones regulatorias en el sistema. Esto puede 
reflejar cuán fuerte o diversa es la influencia de cada componente en la red de expresión génica, lo que podría 
relacionarse con la capacidad del sistema para responder o adaptarse a diversas condiciones.

La suma de los valores absolutos se utiliza como una aproximación simple para capturar la "fuerza" general de las 
interacciones reguladoras, sin que las señales de activación e inhibición se cancelen entre sí. Es decir, cada 
parámetro indica la magnitud de la influencia (ya sea positiva o negativa) que tiene un factor en la red, y al tomar 
el valor absoluto se mide el esfuerzo regulador total.

Esta métrica asume que:
Los parámetros individuales representan contribuciones acumulativas y pueden interactuar de manera no lineal para 
modular la expresión génica.
Valores altos en magnitud pueden estar asociados a respuestas más intensas o complejas, mientras que valores bajos 
indican una regulación más sutil.

En cuanto a las condiciones a las que el sistema podría responder o adaptarse, se consideran escenarios como:
Variaciones ambientales: cambios en condiciones externas (temperatura, señalización hormonal, estrés osmótico) que 
requieren ajustes en la expresión génica.
Señales internas de estrés o daño: respuestas a estrés oxidativo, DNA damage, etc.
Plasticidad fenotípica: la capacidad de adaptarse a nuevos desafíos mediante la modulación de los niveles de proteínas.

l gráfico Complexity-Fitness revela la relación entre esta "complejidad reguladora" y el rendimiento (fitness) de cada 
individuo. Una interpretación posible es:
Baja complejidad y bajo fitness: puede indicar una regulación insuficiente para alcanzar la respuesta óptima.
Alta complejidad y fitness elevado: sugiere que una mayor intensidad en las interacciones puede conducir a una 
respuesta más eficaz, aunque si la complejidad es excesiva, podría reflejar sobre-regulación o inestabilidad.
Tendencias o trade-offs: identificación de un punto óptimo en el cual el balance entre la intensidad de la regulación 
y el desempeño alcanza valores favorables.
En resumen, la métrica y su gráfico ayudan a explorar cómo la suma total de las influencias (sin cancelación de signos) 
se relaciona con la capacidad del sistema para alcanzar un rendimiento óptimo en condiciones variadas.
"""
"""
# Visualización con matplotlib.scatter
plt.figure(figsize=(10, 6))
plt.scatter(complexity, fitness, color="blue", alpha=0.7, edgecolor="k")
plt.xlabel("Complejidad (Suma de |params|)")
plt.ylabel("Fitness")
plt.title("Dispersión: Fitness vs Complejidad")
plt.grid(True)
plt.tight_layout()
# plt.savefig("fitness_vs_complexity.png", dpi=300)
plt.show()
"""
# Alternativa: Visualización con seaborn.scatterplot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=complexity, y=fitness, color="darkred", s=100, edgecolor="w")
plt.xlabel("Complejidad (Suma de |params|)")
plt.ylabel("Fitness")
plt.title("Dispersión: Fitness vs Complejidad")
plt.tight_layout()
plt.show()
