import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Directorio donde se guardan los snapshots (asegúrate que coincide con la configuración de run_demo.py)
snapshot_dir = "snapshots"

# Buscar archivos snapshot que sigan el patrón: snapshot_gen_*.json
snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_gen_*.json")))

if not snapshot_files:
    raise ValueError("No se encontraron archivos snapshot_gen_*.json en la carpeta 'snapshots'.")

# Aquí construiremos una matriz en la que cada fila corresponde a una generación
# y cada columna a la media de cada parámetro (calculado a partir de pop_params de la generación).
parameter_matrix = []

for file in snapshot_files:
    with open(file, "r") as f:
        data = json.load(f)
    # Se asume que cada snapshot tiene la clave "pop_params" que es una lista de listas,
    # donde cada lista interna representa los parámetros de un individuo.
    pop_params = data.get("pop_params", None)
    if pop_params is None:
        continue  # Saltar si no se encontró pop_params en el snapshot
    pop_params = np.array(pop_params)
    # Calcular el promedio de cada parámetro a lo largo de la población
    avg_params = pop_params.mean(axis=0)
    parameter_matrix.append(avg_params)

parameter_matrix = np.array(parameter_matrix)
# """
# Crear el heatmap usando seaborn
plt.figure(figsize=(12, 8))
# cmap=[coolwarm, viridis, plasma, inferno, magma, cividis], annot=[True, False]
# Para un estudio científico, se recomienda usar cmap="magma" o "viridis" y annot=False
heatmap = sns.heatmap(parameter_matrix, cmap="viridis", annot=False, fmt=".2f")
plt.xlabel("Índice del Parámetro")
plt.ylabel("Generación")
plt.title("Heatmap de Evolución de Parámetros por Generación")
plt.tight_layout()
plt.show()
# """
"""
# Crear el heatmap usando plotly
fig = px.imshow(parameter_matrix, 
                labels=dict(x="Índice del Parámetro", y="Generación", color="Media valor"),
                x=[f"param_{i}" for i in range(parameter_matrix.shape[1])],
                y=[f"Gen {i}" for i in range(parameter_matrix.shape[0])],
                color_continuous_scale="Viridis",
                text_auto=True)
fig.update_xaxes(side="top")
"""
# Guardar la figura si es necesario
# plt.savefig("parameter_heatmap.png", dpi=600)
"""
# Mostrar el gráfico interactivo guardándolo en un archivo HTML
output_path = "snapshots/heatmap.html"
fig.write_html(output_path)

print(f"Gráfico guardado en: {os.path.abspath(output_path)}")

# Mostrar el gráfico interactivo
fig.show()

"""