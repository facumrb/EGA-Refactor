import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# -----------------------------
# Paso 1: Cargar la solución final y la configuración
# -----------------------------
# Cargar el archivo final_result.json (se genera en snapshots al terminar run_demo.py)
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
# Se asume que y_values es una lista de listas y tomamos el estado final (última lista)
y_final = y_values[-1]
y_final = np.array(y_final)

# Cargar la configuración para obtener el target. Se usa el archivo config.yaml
config_path = os.path.join("config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
target = config.get("evaluator_params", {}).get("target", None)
if target is None:
    print("No se encontró el target en la configuración.")
    exit(1)
target = np.array(target)

# Definir etiquetas para cada proteína (asumiendo 3 proteínas)
proteins = ["Proteína 1", "Proteína 2", "Proteína 3"]

# -----------------------------
# Versión A: Visualización estática con matplotlib / seaborn
# -----------------------------
fig, ax = plt.subplots(figsize=(8,6))
index = np.arange(len(proteins))
bar_width = 0.35

# Barras para el estado simulado y el target
bars1 = ax.bar(index, y_final, bar_width, label="Simulado", color="skyblue")
bars2 = ax.bar(index + bar_width, target, bar_width, label="Target", color="salmon")

ax.set_xlabel("Proteínas")
ax.set_ylabel("Concentración")
ax.set_title("Comparación entre estado final y target")
ax.set_xticks(index + bar_width/2)
ax.set_xticklabels(proteins)
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# -----------------------------
# Versión B: Visualización interactiva con Plotly
# -----------------------------
fig_interactive = go.Figure()

# Añadir barras para el estado simulado
fig_interactive.add_trace(go.Bar(
    x=proteins,
    y=y_final,
    name="Simulado",
    marker_color="skyblue",
    hovertemplate="Simulado: %{y}<extra></extra>"
))
# Añadir barras para el target
fig_interactive.add_trace(go.Bar(
    x=proteins,
    y=target,
    name="Target",
    marker_color="salmon",
    hovertemplate="Target: %{y}<extra></extra>"
))

fig_interactive.update_layout(
    title="Comparación entre estado final y target",
    xaxis_title="Proteínas",
    yaxis_title="Concentración",
    barmode="group"
)

fig_interactive.show()

# Exportar el gráfico interactivo a HTML (opcional)
# fig_interactive.write_html("comparacion_target_interactivo.html")