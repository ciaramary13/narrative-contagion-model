import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..signals.mispricing import PHASE_COLORS, PHASE_LABELS
from ..model.regime import REGIME_LABELS, REGIME_COLORS


def make_dashboard(
    df: pd.DataFrame,
    sir: pd.DataFrame,
    phases: list[str],
    regimes: np.ndarray,
    mispricing: pd.Series,
) -> go.Figure:
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Price & Narrative Phase",
            "SIR Contagion Dynamics",
            "Mispricing Score",
            "Basic Reproduction Number R₀(t) = β/γ",
            "Market Regime (HMM)",
        ],
        row_heights=[0.30, 0.20, 0.17, 0.17, 0.16],
    )

    index = df.index

    _add_phase_bands(fig, index, phases, row=1)
    fig.add_trace(go.Scatter(
        x=index, y=df["close"],
        line=dict(color="#212121", width=1.5),
        name="Price",
        showlegend=False,
    ), row=1, col=1)

    for col_name, color, label in [
        ("S", "#2196F3", "Susceptible"),
        ("I", "#F44336", "Infected"),
        ("R", "#4CAF50", "Recovered"),
    ]:
        fig.add_trace(go.Scatter(
            x=index,
            y=sir[col_name],
            line=dict(color=color, width=1.5),
            name=label,
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=index,
        y=mispricing,
        line=dict(color="#7B1FA2", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(123, 31, 162, 0.12)",
        name="Mispricing",
        showlegend=False,
    ), row=3, col=1)
    for threshold, color in [(0.5, "rgba(244,67,54,0.5)"), (-0.5, "rgba(33,150,243,0.5)")]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=color, row=3, col=1)

    # R₀(t) = β(t)/γ(t): narrative spreading when > 1, declining when < 1
    fig.add_trace(go.Scatter(
        x=index,
        y=sir["r0"],
        line=dict(color="#FF6F00", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 111, 0, 0.08)",
        name="R₀(t)",
        showlegend=False,
    ), row=4, col=1)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="rgba(0,0,0,0.4)",
        line_width=1,
        row=4, col=1,
    )

    _add_regime_bands(fig, index, regimes, row=5)

    fig.update_layout(
        height=980,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F5F5F5", zeroline=False)

    return fig


def make_phase_legend() -> go.Figure:
    fig = go.Figure()
    for phase, label in PHASE_LABELS.items():
        fig.add_trace(go.Bar(
            x=[label],
            y=[1],
            marker_color=PHASE_COLORS[phase],
            name=label,
            showlegend=False,
            text=label,
            textposition="inside",
        ))
    fig.update_layout(
        height=80,
        margin=dict(t=0, b=0, l=0, r=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        bargap=0.05,
    )
    return fig


def _add_phase_bands(fig: go.Figure, index: pd.Index, phases: list[str], row: int) -> None:
    if not phases:
        return
    current_phase = phases[0]
    start = index[0]

    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            fig.add_vrect(
                x0=start, x1=index[i],
                fillcolor=PHASE_COLORS.get(current_phase, "#EEEEEE"),
                opacity=0.18,
                layer="below",
                line_width=0,
                row=row, col=1,
            )
            current_phase = phases[i]
            start = index[i]

    fig.add_vrect(
        x0=start, x1=index[-1],
        fillcolor=PHASE_COLORS.get(current_phase, "#EEEEEE"),
        opacity=0.18,
        layer="below",
        line_width=0,
        row=row, col=1,
    )


def _add_regime_bands(fig: go.Figure, index: pd.Index, regimes: np.ndarray, row: int) -> None:
    if len(regimes) == 0:
        return
    current = int(regimes[0])
    start = index[0]

    for i in range(1, len(regimes)):
        r = int(regimes[i])
        if r != current:
            fig.add_vrect(
                x0=start, x1=index[i],
                fillcolor=REGIME_COLORS.get(current, "#EEEEEE"),
                opacity=0.25,
                layer="below",
                line_width=0,
                row=row, col=1,
            )
            current = r
            start = index[i]

    fig.add_vrect(
        x0=start, x1=index[-1],
        fillcolor=REGIME_COLORS.get(current, "#EEEEEE"),
        opacity=0.25,
        layer="below",
        line_width=0,
        row=row, col=1,
    )

    for state_id, label in REGIME_LABELS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=REGIME_COLORS[state_id], size=10, symbol="square"),
            name=f"Regime: {label}",
        ), row=row, col=1)
