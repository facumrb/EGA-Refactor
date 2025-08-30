import os
import json
import numpy as np
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

# Definir umbral de "complejidad aceptable".
# En este ejemplo usamos la mediana como referencia.
acceptable_threshold = np.median(complexity)

# Seleccionar entre los individuos con complejidad <= acceptable_threshold el que tenga el fitness mínimo
eligible_indices = [i for i, comp in enumerate(complexity) if comp <= acceptable_threshold]
if eligible_indices:
    best_idx = min(eligible_indices, key=lambda i: fitness[i])
else:
    best_idx = np.argmin(fitness)

""" REVISAR:
La "complejidad", definida como la suma de los valores absolutos de los parámetros, puede interpretarse biológicamente 
como una medida global de la intensidad o magnitud de las interacciones regulatorias en el sistema. Este valor refleja 
cuán fuerte o diversa es la influencia de cada componente en la red de expresión génica, lo que podría relacionarse 
con la capacidad del sistema para responder a condiciones variables.

La suma de los valores absolutos se utiliza como una aproximación simple para capturar la "fuerza" general de las 
interacciones reguladoras, evitando que las señales activadoras e inhibitorias se cancelen mutuamente. Es decir, cada 
parámetro indica la magnitud de la influencia (positiva o negativa) de un factor en la red, y su valor absoluto 
cuantifica el esfuerzo regulador total.

Esta métrica asume que:
Los parámetros individuales representan contribuciones acumulativas que, aunque puedan interactuar de forma no lineal, 
se suman para modular la expresión génica.
Valores altos en magnitud pueden estar asociados a respuestas más intensas o complejas, mientras que valores bajos 
indican una regulación más sutil.

En cuanto a las condiciones a las que el sistema podría responder o adaptarse, se consideran escenarios como:
Variaciones ambientales: cambios en condiciones externas (temperatura, señalización hormonal, estrés osmótico) que 
requieren ajustes en la expresión génica.
Señales internas de estrés o daño: respuestas a estrés oxidativo, daño en el DNA, etc.
Plasticidad fenotípica: la capacidad del sistema para adaptarse a nuevos desafíos modulando los niveles de proteínas.

El gráfico Complexity-Fitness revela la relación entre esta "complejidad reguladora" y el rendimiento (fitness) de cada 
individuo. Una interpretación posible es:
Baja complejidad y bajo fitness: puede indicar que el sistema tiene la regulación adecuada para acercarse al objetivo 
sin excesos.
Alta complejidad y fitness elevado: sugiere que una intensidad regulatoria excesiva puede conducir a un desempeño 
ineficaz, posiblemente por sobre-regulación o inestabilidad.
Tendencias o trade-offs: la identificación de un punto óptimo en el que el balance entre la intensidad regulatoria y el desempeño es favorable.

En resumen, esta métrica y su gráfico permiten explorar cómo la suma total de las influencias regulatorias (sin cancelación de signos) se 
relaciona con la capacidad del sistema para alcanzar un rendimiento óptimo bajo condiciones variadas.
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
# plt.savefig("fitness_vs_complexity.png", dpi=600)
plt.show()
"""
# Alternativa: Visualización con seaborn.scatterplot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=complexity, y=fitness, color="darkred", s=100, edgecolor="w", label="Individuos")

# Resaltar el punto de menor fitness (mejor) con complejidad aceptable
plt.scatter([complexity[best_idx]], [fitness[best_idx]], color="limegreen", s=150, edgecolor="black", zorder=5, label="Mejor (aceptable)")

plt.xlabel("Complejidad: suma de la influencia de cada componente en la red de expresión génica")
plt.ylabel("Fitness")
plt.title("Dispersión: Fitness vs Complejidad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
