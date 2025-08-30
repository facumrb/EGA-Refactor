import os
import json
from typing import final
import yaml
import numpy as np
import networkx as nx
import plotly.graph_objects as go

# -----------------------------
# Paso 1: Cargar datos
# -----------------------------

# Cargar la configuración para saber el número de proteínas
with open("config.yaml", "r") as filehandler:
    config = yaml.safe_load(filehandler)
num_proteins = len(config.get("evaluator_params", {}).get("target", []))
if num_proteins == 0:
    print("Error: No se pudo determinar el número de proteínas desde config.yaml")
    exit(1)

# Cargar el resultado final para obtener el mejor individuo
with open("snapshots/final_result.json", "r") as filehandler:
    final_result = json.load(filehandler)
best_individual_params = final_result.get("best", {}).get("params", [])
if not best_individual_params:
    print("Error: No se encontró el cromosoma del mejor individuo en final_result.json")
    exit(1)
best_individual_params = final_result['best']['params']
num_proteins = len(config['evaluator_params']['target'])

# -----------------------------
# Paso 2: Procesar el genoma para obtener la matriz de interacciones
# -----------------------------
"""
Estas proteínas, a su vez, pueden actuar como "interruptores" para otros genes, activando o inhibiendo su expresión. 
Esto crea redes complejas donde los genes se controlan entre sí para llevar a cabo funciones celulares complejas, 
como el desarrollo de un tejido o la respuesta a un estímulo.
"""

# El genoma tiene el formato: [prod_1, deg_1, inter_1, prod_2, deg_2, inter_2, ...]
production_rates = best_individual_params[0::3]
degradation_rates = best_individual_params[1::3]
interaction_weights = best_individual_params[2::3]
num_weights_found = len(interaction_weights)
num_weights_expected = num_proteins
"""
Cambio de Concentración = Tasa de Producción - Tasa de Degradación
1. Producción : Se asume que cada gen tiene una tasa de producción base. La parte crucial es el término de regulación: 
el modelo actual simplifica enormemente este proceso. En lugar de que cada proteína influya individualmente en las 
demás, asume que la suma total de las concentraciones de todas las proteínas influye en la producción de cada gen a 
través de un único "peso de interacción". Un peso positivo simula una activación (más proteínas en total -> mayor 
producción) y uno negativo una inhibición .
2. Degradación : Se modela como una tasa de decaimiento simple, donde la cantidad de proteína que se degrada es 
proporcional a la cantidad que ya existe. Esta es una aproximación estándar y bastante razonable en muchos modelos 
biológicos.
"""
# Validar que tengamos el número correcto de pesos
if num_weights_found != num_weights_expected:
    raise ValueError(f"Error: Se esperaban {num_weights_expected} pesos de interacción, pero se encontraron {num_weights_found}")

# -----------------------------
# Paso 3: Crear el grafo con NetworkX
# -----------------------------
# Crear un grafo dirigido
G = nx.DiGraph() # Usamos un grafo dirigido para mostrar la dirección de la influencia

# Añadir nodos (proteínas)
protein_names = [f"P{i+1}" for i in range(num_proteins)]
central_node_name = "Influencia Global"
G.add_nodes_from(protein_names)
G.add_node(central_node_name)

# Conectar las proteínas al nodo central y desde el nodo central a las proteínas
for i, protein in enumerate(protein_names):
    # Flecha desde la proteína hacia el nodo central (simboliza contribución)
    G.add_edge(protein, central_node_name, weight=1, type='contribution') 
    # Flecha desde el nodo central a la proteína con el peso de interacción
    G.add_edge(central_node_name, protein, weight=interaction_weights[i], type='regulation')

# Posiciones de los nodos para la visualización (colocamos el nodo central en el medio)
pos = nx.spring_layout(G, seed=config['ega_params']['seed'])
pos[central_node_name] = np.array([0, 0]) # Forzar el nodo central al origen

# -----------------------------
# Paso 4: Visualizar el grafo con Plotly
# -----------------------------

# --- Creación de las aristas (flechas) ---
edge_traces = []
for u, v, data in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    weight = data.get('weight', 0)
    edge_type = data.get('type', '')

    # Personalización visual según el tipo de borde
    if edge_type == 'regulation':
        color = 'green' if weight > 0 else 'red'
        # Normalizar el grosor de la línea basado en el máximo peso absoluto
        max_abs_weight = max(abs(w) for w in interaction_weights) if interaction_weights else 1
        width = 1 + 4 * abs(weight) / max_abs_weight
        hover_text = f'Peso Regulación: {weight:.2f}'
    else: # 'contribution'
        color = 'grey'
        width = 0.5
        hover_text = 'Contribuye a la influencia total'

    edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color, dash='dot' if edge_type == 'contribution' else 'solid'),
            hoverinfo='text',
            text=hover_text,
            mode='lines')
    edge_traces.append(edge_trace)

# --- Creación de los nodos ---
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_sizes = [40 if node == central_node_name else 25 for node in G.nodes()]
node_colors = ['lightblue' if node == central_node_name else 'lightpink' for node in G.nodes()]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=[node for node in G.nodes()],
    textposition="bottom center",
    marker=dict(
        showscale=False, # Desactivado para usar colores directos
        color=node_colors,
        size=node_sizes,
        line_width=2))

# --- Añadir información a los nodos y aristas para el hover ---
node_text = []
for node in G.nodes():
    if node == central_node_name:
        node_text.append("Este nodo representa la suma de las concentraciones de todas las proteínas, que influye en la tasa de producción de cada gen.")
    else:
        # Encontrar el índice de la proteína para acceder a sus tasas
        protein_index = protein_names.index(node)
        prod_rate = production_rates[protein_index]
        deg_rate = degradation_rates[protein_index]
        # Encontrar el peso de la arista que viene del nodo central
        in_weight = G.get_edge_data(central_node_name, node).get('weight', 0)
        
        node_text.append(f'<b>Proteína: {node}</b><br>'+
                         f'Tasa de Producción: {prod_rate:.2f}<br>'+
                         f'Tasa de Degradación: {deg_rate:.2f}<br>'+
                         f'Regulada por un peso de: {in_weight:.2f}')

node_trace.hovertext = node_text # Usar hovertext en lugar de text para el hover

# --- Creación de la figura ---
fig = go.Figure(data=[*edge_traces, node_trace],
             layout=go.Layout(
                title='<br>Red de Interacción Génica con Tasas de Producción/Degradación', 
                title_font_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Pasa el cursor sobre los nodos para ver sus tasas y pesos de regulación.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )

# Guardar y mostrar
output_path = "./gene_network.html"
fig.write_html(output_path)
print(f"Grafo de la red génica guardado en: {os.path.abspath(output_path)}")