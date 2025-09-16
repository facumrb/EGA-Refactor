import json
import os

import plotly.graph_objects as go
import numpy as np

def plot_3d_trajectory(file_path="../snapshots/final_result.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Asumiendo que 'y' y 't' están en el primer elemento de 'population'
    # y que 'y' es una lista de listas donde cada sublista es una proteína a lo largo del tiempo
    # y que 't' es el tiempo correspondiente
    solution_y = data['population'][0]['y']
    solution_t = data['population'][0]['t']

    if solution_y is None or solution_t is None:
        print("No se encontraron datos de trayectoria (solution.y o solution.t) en el archivo.")
        return

    # Convertir a array de numpy para facilitar el acceso
    solution_y = np.array(solution_y)

    # Asegurarse de que tenemos al menos 3 proteínas para una visualización 3D
    if solution_y.shape[0] < 3:
        print("Se necesitan al menos 3 proteínas para una visualización 3D.")
        return

    # Crear la figura 3D
    fig = go.Figure(data=[go.Scatter3d(
        x=solution_y[0, :],
        y=solution_y[1, :],
        z=solution_y[2, :],
        mode='lines+markers',
        marker=dict(
            size=2,
            color=solution_t, # Color por tiempo
            colorscale='Viridis',
            colorbar=dict(title='Tiempo'),
            line=dict(width=0)
        ),
        line=dict(color='darkblue', width=2)
    )])

    # Actualizar el layout
    fig.update_layout(
        title='Trayectoria en Espacio de Fases 3D (Proteínas P1, P2, P3)',
        scene=dict(
            xaxis_title='Concentración Proteína 1',
            yaxis_title='Concentración Proteína 2',
            zaxis_title='Concentración Proteína 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    if not os.path.exists("./visuals"):
        os.makedirs("./visuals")
    # Guardar la figura como HTML
    fig.write_html("./visuals/3d_trajectory.html", auto_open=True)
    
if __name__ == "__main__":
    plot_3d_trajectory()