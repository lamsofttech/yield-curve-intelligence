"""Plotly figure builders. Deliberately UI-agnostic (return go.Figure
objects, take no Gradio-specific arguments) so each is testable with plain
pytest without a running Gradio app -- app.py only wires these into
components, it contains no chart-construction logic itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from interpolation import METHODS


def plot_method_comparison(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    target_maturity: float | None = None,
    n_grid: int = 400,
) -> go.Figure:
    """One curve per interpolation method over a dense grid, observed
    points marked separately, and the target maturity (if given) marked
    with a vertical line -- so a user can see exactly where methods agree
    and where they diverge.
    """
    grid = np.linspace(x_nodes.min(), x_nodes.max(), n_grid)
    fig = go.Figure()
    for name, method in METHODS.items():
        try:
            values = method(x_nodes, y_nodes, grid)
        except ValueError:
            continue  # skip a method that can't handle these nodes, rather than crash the chart
        fig.add_trace(go.Scatter(x=grid, y=values, mode="lines", name=name))
    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers",
            name="Observed",
            marker=dict(color="black", size=9, symbol="circle"),
        )
    )
    if target_maturity is not None:
        fig.add_vline(x=target_maturity, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Interpolation method comparison",
        xaxis_title="Maturity (years)",
        yaxis_title="Yield (%)",
        legend_title="Method",
    )
    return fig


def plot_yield_curve(
    x_nodes: np.ndarray, y_nodes: np.ndarray, title: str = "Yield curve"
) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x_nodes, y=y_nodes, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title="Maturity (years)", yaxis_title="Yield (%)")
    return fig


def plot_loo_errors(results) -> go.Figure:
    """Bar chart of MAE per method. Log-scale y-axis: Lagrange's leave-one-
    out error on the full grid is orders of magnitude larger than the
    local methods' (see DECISIONS.md) -- a linear axis would render the
    other three bars invisibly flat by comparison.
    """
    names = [r.method for r in results]
    maes = [max(r.mae, 1e-6) for r in results]  # floor to avoid log(0)
    fig = go.Figure(go.Bar(x=names, y=maes))
    fig.update_layout(
        title="Leave-one-out mean absolute error by method (log scale)",
        yaxis_title="MAE (percentage points)",
        yaxis_type="log",
    )
    return fig


def plot_pca_loadings(loadings: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in loadings.columns:
        fig.add_trace(
            go.Scatter(x=loadings.index, y=loadings[column], mode="lines+markers", name=column)
        )
    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    fig.update_layout(
        title="PCA loadings by maturity",
        xaxis_title="Maturity (years)",
        yaxis_title="Loading",
        legend_title="Component",
    )
    return fig


def plot_pca_scores(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in scores.columns:
        fig.add_trace(go.Scatter(x=scores.index, y=scores[column], mode="lines", name=column))
    fig.update_layout(
        title="PCA factor scores over time",
        xaxis_title="Date",
        yaxis_title="Score",
        legend_title="Component",
    )
    return fig


def plot_anomaly_scores(scores: pd.Series, is_anomaly: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores.index, y=scores.values, mode="lines", name="Anomaly score"))
    flagged = scores[is_anomaly]
    fig.add_trace(
        go.Scatter(
            x=flagged.index,
            y=flagged.values,
            mode="markers",
            name="Flagged as unusual",
            marker=dict(color="red", size=6),
        )
    )
    fig.update_layout(
        title="Curve-shape anomaly score over time (lower = more unusual)",
        xaxis_title="Date",
        yaxis_title="Anomaly score",
    )
    return fig
