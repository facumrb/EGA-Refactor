import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import mpld3
import webbrowser

# Directorio donde se guardan los snapshots (asegúrate que coincide con la configuración de run_demo.py)
snapshot_dir = "../snapshots"

# Buscar archivos snapshot que sigan el patrón: snapshot_gen_*.json
snapshot_files = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_gen_*.json")))

if not snapshot_files:
    raise ValueError("No se encontraron archivos snapshot_gen_*.json en la carpeta 'snapshots'.")

# Aquí construiremos una matriz en la que cada fila corresponde a una generación
# y cada columna a la media de cada parámetro (calculado a partir de pop_params de la generación).
parameter_matrix = []

for file in snapshot_files:
    with open(file, "r") as filehadler:
        data = json.load(filehadler)
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

# Crear el heatmap usando seaborn
plt.figure(figsize=(12, 8))
# cmap=[coolwarm, viridis, plasma, inferno, magma, cividis], annot=[True, False]
# Para un estudio científico, se recomienda usar cmap="magma" o "viridis" y annot=False
heatmap = sns.heatmap(parameter_matrix, cmap="viridis", annot=False, fmt=".2f")
plt.xlabel("Parámetros de producción (1, 4, 7), degradación (2, 5, 8) e interacción (3, 6, 9)")
plt.ylabel("Generación")
plt.title("Heatmap de Evolución de Parámetros por Generación")
# Añade estas líneas para modificar los ticks de los ejes
plt.xticks(ticks=np.arange(0.5, parameter_matrix.shape[1] + 0.5, 1), labels=np.arange(1, parameter_matrix.shape[1] + 1, 1))
plt.yticks(ticks=np.arange(0.5, parameter_matrix.shape[0] + 0.5, 1), labels=np.arange(1, parameter_matrix.shape[0] + 1, 1)[::-1])
plt.tight_layout()
# plt.show()
# Convert matplotlib figure to HTML
html_fig = mpld3.fig_to_html(plt.gcf())

# Save HTML to file and open in default browser
output_path = "visuals/heatmap.html"
with open(output_path, "w") as f:
    f.write(html_fig)
webbrowser.open('file://' + os.path.realpath(output_path))

# Guardar la figura si es necesario
# plt.savefig("./visuals/heatmap.png", dpi=600)


"""
# Para el heatmap de plotly (versión interactiva):
fig = px.imshow(parameter_matrix, 
                labels=dict(x="Índice del Parámetro", y="Generación", color="Media valor"),
                x=[str(i) for i in range(1, parameter_matrix.shape[1]+1)],  # Eje X de 1 a 9
                y=[str(i) for i in range(1, parameter_matrix.shape[0]+1)][::-1],  # Eje Y de 1 a 80
                color_continuous_scale="Viridis",
                text_auto=True)
fig.update_xaxes(side="top")

fig.update_layout(
    autosize=True, # Habilita el ajuste automático
    width=None,    # Permite que el ancho se ajuste automáticamente
    height=None,   # Permite que la altura se ajuste automáticamente
    margin=dict(l=50, r=50, b=50, t=50)  # Ajusta los márgenes si es necesario
)

# Mostrar el gráfico interactivo guardándolo en un archivo HTML
output_path = "visuals/heatmap.html"
fig.write_html(output_path)

print(f"Gráfico guardado en: {os.path.abspath(output_path)}")

# Mostrar el gráfico interactivo
fig.show()

"""