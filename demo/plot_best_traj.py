import json
import os
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------
# Paso 1. Cargar datos de la solución
# -----------------------------
# Se asume que el archivo final_result.json se encuentra en la carpeta snapshots.
snapshot_dir = "snapshots"
final_result_path = os.path.join(snapshot_dir, "final_result.json")

with open(final_result_path, "r") as f:
    final_result = json.load(f)

# La solución del mejor individuo se espera que esté en la clave "best" del JSON.
# Debe incluir "t" (vector de tiempos) y "y" (matriz de concentraciones, cada columna para una proteína).
best_solution = final_result.get("best", {})
time = best_solution.get("t", [])
y_values = best_solution.get("y", [])

if not time or not y_values:
    print("No se encontraron datos de la trayectoria en la solución.")
    exit(1)

# Suponiendo que y_values es una lista de listas con 3 elementos (p. ej., [[p1, p2, p3], ...])
protein1 = y_values[0]
protein2 = y_values[1]
protein3 = y_values[2]

# -----------------------------
# Opción 1: Visualización estática con matplotlib
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(time, protein1, label="Proteína 1")
plt.plot(time, protein2, label="Proteína 2")
plt.plot(time, protein3, label="Proteína 3")
plt.xlabel("Tiempo")
plt.ylabel("Concentración de las proteínas (TFs)")
plt.title("Trayectoria temporal del mejor individuo")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("trayectoria_mejor_individuo.png", dpi=600)
plt.show()
"""
# -----------------------------
# Opción 2: Visualización interactiva con Plotly
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=protein1,
                         mode="lines+markers", name="Proteína 1"))
fig.add_trace(go.Scatter(x=time, y=protein2,
                         mode="lines+markers", name="Proteína 2"))
fig.add_trace(go.Scatter(x=time, y=protein3,
                         mode="lines+markers", name="Proteína 3"))
fig.update_layout(title="Trayectoria temporal del mejor individuo",
                  xaxis_title="Tiempo",
                  yaxis_title="Concentración de las proteínas (TFs)",
                  hovermode="x unified")

# Agregar menú interactivo para filtrar las trazas de proteínas
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            buttons=list([
                dict(
                    label="Mostrar Todas",
                    method="update",
                    args=[{"visible": [True, True, True]},
                          {"title": "Todas las Proteínas"}]
                ),
                dict(
                    label="Proteína 1",
                    method="update",
                    args=[{"visible": [True, False, False]},
                          {"title": "Solo Proteína 1"}]
                ),
                dict(
                    label="Proteína 2",
                    method="update",
                    args=[{"visible": [False, True, False]},
                          {"title": "Solo Proteína 2"}]
                ),
                dict(
                    label="Proteína 3",
                    method="update",
                    args=[{"visible": [False, False, True]},
                          {"title": "Solo Proteína 3"}]
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.15,
            xanchor="right",
            y=0.5,
            yanchor="middle"
        )
    ]
)

# Mostrar el gráfico interactivo
# Suprimir la salida de JS de Plotly redirigiendo stderr temporalmente
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    fig.show()
finally:
    sys.stderr.close()
    sys.stderr = original_stderr


# Exportar el gráfico interactivo a un archivo HTML
fig.write_html("trayectoria_interactiva.html")
"""