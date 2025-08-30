"""
Este script carga los resultados finales (snapshots) generados por run_demo.py y
genera dos visualizaciones del spaghetti plot:
    - Opción A: Gráfico estático con Matplotlib (líneas traslúcidas).
    - Opción B: Gráfico interactivo con Plotly.
Se asume que en final_result.json (dentro de la carpeta snapshots) existe el campo "spaghetti_results",
donde cada entrada es un dict con dos claves: "t" (puntos temporales) y "y" (trayectorias de proteínas, 
una lista de listas, una por proteína).
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Carga y validación de datos ---
# Ruta al archivo final_result.json dentro de la carpeta snapshots
snapshot_path = os.path.join("snapshots", "final_result.json")
if not os.path.exists(snapshot_path):
    raise FileNotFoundError(f"No se encontró el archivo de resultados en: {snapshot_path}")
    sys.exit(1)

# Cargar el archivo JSON
with open(snapshot_path, "r") as filehandler:
    result = json.load(filehandler)

# Extraer y validar los resultados de spaghetti
spaghetti_data = result.get("spaghetti_results")
if not spaghetti_data or not isinstance(spaghetti_data, dict):
    raise ValueError("El campo 'spaghetti_results' no se encontró o es inválido en final_result.json.")

simulations = spaghetti_data.get("simulations")
if not simulations or not isinstance(simulations, list) or not simulations:
    raise ValueError("No se encontraron simulaciones ('simulations') válidas en 'spaghetti_results'.")

# Determinar el número de proteínas a partir de la primera simulación
try:
    num_proteins = len(simulations[0]["y"])
except (KeyError, IndexError, TypeError):
    raise ValueError("No se pudo determinar el número de proteínas a partir de los datos de la simulación.")

# Generar un colormap para distinguir proteínas
colors = plt.cm.viridis(np.linspace(0, 1, num_proteins))

# ------------------------------
# Opción A: Matplotlib con líneas traslúcidas
# ------------------------------
plt.figure(figsize=(12, 8))

# Usar el primer resultado para determinar el número de proteínas.
try:
    first_sim = simulations[0]
    if "y" not in first_sim or not first_sim["y"]:
        raise ValueError("La primera simulación no contiene trayectorias 'y'.")
    num_proteins = len(first_sim["y"])
except (IndexError, KeyError, TypeError) as e:
    raise ValueError(f"Error al procesar la primera simulación: {e}")

# Iterar cada simulación y trazar cada proteína
for sim in simulations:
    t = sim.get("t")
    y = sim.get("y")
    if t is not None and y is not None and len(y) == num_proteins:
        for idx, protein_traj in enumerate(y):
            plt.plot(t, protein_traj, color=colors[idx], alpha=0.3)

# Crear líneas ficticias para la leyenda
for idx in range(num_proteins):
    plt.plot([], [], color=colors[idx], label=f'Proteína {idx+1}')

plt.xlabel("Tiempo")
plt.ylabel("Concentración de las proteínas (TFs)")
plt.title("Spaghetti Plot: Simulaciones ruidosas del mejor individuo")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("trayectoria_mejor_individuo.png", dpi=600)
plt.show()

"""
# ------------------------------
# Opción B: Plotly para interactividad
# ------------------------------
fig = go.Figure()

# Colores para Plotly (formato 'rgba(r,g,b,a)')
plotly_colors = [f'rgba({r*255},{g*255},{b*255},1)' for r, g, b, a in colors]

# Cada proteína tendrá sus trazos de cada simulación.
for protein_idx in range(num_proteins):
    # Añadir una traza por simulación para esta proteína
    for i, sim in enumerate(simulations):
        t = sim.get("t")
        y = sim.get("y")
        if t is not None and y is not None and len(y) == num_proteins:
            fig.add_trace(go.Scatter(
                x=t,
                y=y[protein_idx],
                mode="lines",
                line=dict(width=1, color=plotly_colors[protein_idx]), # width=1.5
                opacity=0.3, # opacity=0.4
                name=f"Proteína {protein_idx+1}",
                legendgroup=f"Proteína {protein_idx+1}",
                showlegend=(i == 0) # Mostrar leyenda solo para la primera simulación de cada proteína
            ))

# Configuramos el layout
fig.update_layout(
    title="Spaghetti Plot interactivo: Simulaciones ruidosas del mejor individuo",
    xaxis_title="Tiempo",
    yaxis_title="Concentración de las proteínas (TFs)"
)

# Redirigir stderr para suprimir la salida de JS de Plotly en la consola
# Esto es opcional pero mejora la limpieza de la salida.
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    fig.show()
finally:
    sys.stderr.close()
    sys.stderr = original_stderr

# Exportar el gráfico interactivo a un archivo HTML
# fig.write_html("spaghetti_interactivo.html")
"""