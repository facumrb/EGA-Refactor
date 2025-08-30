import os
import json
import yaml
import numpy as np
import networkx as nx
import plotly.graph_objects as go

def load_configuration():
    """Carga la configuración desde config.yaml"""
    with open("config.yaml", "r") as filehandler:
        config = yaml.safe_load(filehandler)
    num_proteins = len(config.get("evaluator_params", {}).get("target", []))
    if num_proteins == 0:
        raise ValueError("No se pudo determinar el número de proteínas desde config.yaml")
    return config, num_proteins

def load_snapshot_data():
    """Carga los datos de los snapshots"""
    snapshot_files = sorted([
        f for f in os.listdir('snapshots') 
        if f.startswith('snapshot_gen_') and f.endswith('.json')
    ])
    snapshots = []
    for snapshot_file in snapshot_files:
        with open(f'snapshots/{snapshot_file}', 'r') as f:
            snapshots.append(json.load(f))
    return snapshots

def create_edge_traces(G, pos, interaction_weights):
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get('weight', 0)
        edge_type = data.get('type', '')

        if edge_type == 'regulation':
            color = 'green' if weight > 0 else 'red'
            max_abs_weight = max(abs(w) for w in interaction_weights) if len(interaction_weights) > 0 else 1
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
    
    return edge_traces

def create_node_trace(G, pos, production_rates, degradation_rates, protein_names, central_node_name):
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_sizes = [40 if node == central_node_name else 25 for node in G.nodes()]
    node_colors = ['lightblue' if node == central_node_name else 'lightpink' for node in G.nodes()]

    node_text = []
    for node in G.nodes():
        if node == central_node_name:
            node_text.append("Este nodo representa la suma de las concentraciones de todas las proteínas, que influye en la tasa de producción de cada gen.")
        else:
            protein_index = protein_names.index(node)
            prod_rate = production_rates[protein_index]
            deg_rate = degradation_rates[protein_index]
            in_weight = G.get_edge_data(central_node_name, node).get('weight', 0)
            
            node_text.append(f'<b>Proteína: {node}</b><br>'+
                             f'Tasa de Producción: {prod_rate:.2f}<br>'+
                             f'Tasa de Degradación: {deg_rate:.2f}<br>'+
                             f'Regulada por un peso de: {in_weight:.2f}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        textposition="bottom center",
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=2))
    
    return node_trace

if __name__ == "__main__":
    # Cargar configuración inicial
    config, num_proteins = load_configuration()
    protein_names = [f"P{i+1}" for i in range(num_proteins)]
    central_node_name = "Influencia Global"
    
    # Cargar datos de los snapshots
    snapshots = load_snapshot_data()
    
    print(f"Cargados {len(snapshots)} snapshots para animación")

    frames = []
    for i, snapshot in enumerate(snapshots):
        generation = snapshot['gen']
        best_individual_params = snapshot['best_params']

        production_rates = best_individual_params[0::3]
        degradation_rates = best_individual_params[1::3]
        interaction_weights = best_individual_params[2::3]

        # Crear el grafo para esta generación
        G = nx.DiGraph()
        G.add_nodes_from(protein_names)
        G.add_node(central_node_name)

        for j, protein in enumerate(protein_names):
            G.add_edge(protein, central_node_name, weight=1, type='contribution')
            G.add_edge(central_node_name, protein, weight=interaction_weights[j], type='regulation')

        pos = nx.spring_layout(G, seed=config['ega_params']['seed'])
        pos[central_node_name] = np.array([0, 0])

        # Crear trazas para esta generación
        edge_traces = create_edge_traces(G, pos, interaction_weights)
        node_trace = create_node_trace(G, pos, production_rates, degradation_rates, protein_names, central_node_name)

        frames.append(go.Frame(data=[*edge_traces, node_trace], name=str(generation)))

    # Crear la figura inicial (usando el primer snapshot)
    initial_snapshot = snapshots[0]
    initial_best_params = initial_snapshot['best_params']
    initial_production_rates = initial_best_params[0::3]
    initial_degradation_rates = initial_best_params[1::3]
    initial_interaction_weights = initial_best_params[2::3]

    initial_G = nx.DiGraph()
    initial_G.add_nodes_from(protein_names)
    initial_G.add_node(central_node_name)
    for j, protein in enumerate(protein_names):
        initial_G.add_edge(protein, central_node_name, weight=1, type='contribution')
        initial_G.add_edge(central_node_name, protein, weight=initial_interaction_weights[j], type='regulation')
    initial_pos = nx.spring_layout(initial_G, seed=config['ega_params']['seed'])
    initial_pos[central_node_name] = np.array([0, 0])

    initial_edge_traces = create_edge_traces(initial_G, initial_pos, initial_interaction_weights)
    initial_node_trace = create_node_trace(initial_G, initial_pos, initial_production_rates, initial_degradation_rates, protein_names, central_node_name)

    fig = go.Figure(
        data=[*initial_edge_traces, initial_node_trace],
        layout=go.Layout(
            title='<br>Red de Interacción Génica (Animación por Generación)',
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
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ),
        frames=frames
    )

    # Guardar la animación en un archivo HTML
    fig.write_html("gene_network_animation.html", auto_open=True)
    print("Animación guardada en gene_network_animation.html")