# -----------------------------
# Versión B: Visualización interactiva con Plotly (mejorada)
# -----------------------------
fig_interactive = go.Figure()

# Añadir barras para el estado simulado con tooltips enriquecidos
fig_interactive.add_trace(go.Bar(
    x=proteins,
    y=y_final,
    name="Simulado",
    marker_color="skyblue",
    hovertemplate="<b>%{x}</b><br>Simulado: %{y:.3f}<extra></extra>"
))

# Añadir barras para el target con tooltips enriquecidos
fig_interactive.add_trace(go.Bar(
    x=proteins,
    y=target,
    name="Target",
    marker_color="salmon",
    hovertemplate="<b>%{x}</b><br>Target: %{y:.3f}<extra></extra>"
))

fig_interactive.update_layout(
    title="Comparación entre estado final y target",
    xaxis_title="Proteínas",
    yaxis_title="Concentración",
    barmode="group",
    hovermode="x"
)

# Mostrar el gráfico interactivo
fig_interactive.show()

# Exportar el gráfico interactivo a HTML para compartir o visualizar en el navegador
fig_interactive.write_html("comparacion_target_interactivo.html")