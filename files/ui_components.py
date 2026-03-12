import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

PLOTLY_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font_color": "#94a3b8",
    "gridcolor": "rgba(255,255,255,0.05)",
}

def plotly_layout(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    """Applique le thème ultra premium transparent à une figure Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=16, color="#f8fafc", weight="bold")),
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(color=PLOTLY_THEME["font_color"], family="Inter, sans-serif"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=20),
        xaxis=dict(gridcolor=PLOTLY_THEME["gridcolor"], showgrid=True, gridwidth=1, zeroline=False),
        yaxis=dict(gridcolor=PLOTLY_THEME["gridcolor"], showgrid=True, gridwidth=1, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_size=13, font_family="Inter"),
    )
    return fig

def metric_card(label: str, value: str, delta: str = "", color: str = "#e2e8f0", tooltip: str = "") -> None:
    delta_html = f'<div class="metric-delta" style="color:{color}">{delta}</div>' if delta else ""
    
    # Intégration du tooltip si fourni
    label_html = f"""
    <div class="tooltip-container">
        {label}
        <span class="tooltip-text">{tooltip}</span>
    </div>
    """ if tooltip else f"{label}"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label_html}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def section_header(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
