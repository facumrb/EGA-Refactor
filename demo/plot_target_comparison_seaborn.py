import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paso 1: Cargar la solución final y la configuración
# -----------------------------
# Cargar el archivo final_result.json
snapshot_dir = "snapshots"
final_result_path = os.path.join(snapshot_dir, "final_result.json")
with open(final_result_path, "r") as f:
    final_result = json.load(f)

# La solución del mejor individuo está en la clave "best"
best_solution = final_result.get("best", {})
y_values = best_solution.get("y", [])
if not y_values:
    print("No se encontraron datos de las concentraciones en la solución.")
    exit(1)

y_final = [y_values[0][-1], y_values[1][-1], y_values[2][-1]]

# Cargar la configuración para obtener el target
config_path = os.path.join("config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
target = config.get("evaluator_params", {}).get("target", None)
if target is None:
    print("No se encontró el target en la configuración.")
    exit(1)
target = np.array(target)

if len(y_final) != len(target):
    print(
        f"Error: La longitud de la solución final ({len(y_final)}) "
        f"no coincide con la del target ({len(target)}).", y_final, target
    )
    exit(1)

num_proteins = len(target)
proteins = [f"Proteína {i+1}" for i in range(num_proteins)]

# -----------------------------
# Versión A: Visualización estática con Seaborn
# -----------------------------
# Crear DataFrame para Seaborn
data = {
    "Proteína": proteins * 2,
    "Concentración": np.concatenate([y_final, target]),
    "Tipo": ["Simulado"] * len(proteins) + ["Target"] * len(proteins)
}
df = pd.DataFrame(data)

# Establecer estilo de Seaborn
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,6))
ax = sns.barplot(x="Proteína", y="Concentración", hue="Tipo", data=df, palette=["skyblue", "salmon"])
ax.set_title("Comparación entre estado final y target")
plt.tight_layout()
plt.show()