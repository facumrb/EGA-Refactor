import json
import os
import matplotlib.pyplot as plt

# Ruta al archivo final_result.json generado en el directorio de snapshots
snapshot_dir = os.path.join("snapshots")
final_result_path = os.path.join(snapshot_dir, "final_result.json")

# Cargar los datos del resultado final
with open(final_result_path, "r") as fh:
    final_result = json.load(fh)

history = final_result.get("history", {})
min_fitness = history.get("min", [])
avg_fitness = history.get("avg", [])

generations = list(range(len(min_fitness)))

# Crear la figura y graficar
plt.figure(figsize=(10, 6))
plt.plot(generations, min_fitness, label="Fitness Mínimo", marker="o")
plt.plot(generations, avg_fitness, label="Fitness Promedio", marker="s")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del Fitness por Generación")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig("fitness_evolution.png", dpi=600)
plt.show()