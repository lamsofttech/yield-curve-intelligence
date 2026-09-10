import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.loo_validation import validate_all_methods
from visualization.curve_plots import (
    plot_anomaly_scores,
    plot_loo_errors,
    plot_method_comparison,
    plot_pca_loadings,
    plot_pca_scores,
    plot_yield_curve,
)

X = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
Y = np.array([4.15, 4.39, 4.44, 4.57, 4.68, 4.80])


def test_plot_method_comparison_has_one_trace_per_method_plus_observed():
    fig = plot_method_comparison(X, Y, target_maturity=4.0)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "Observed" in trace_names
    assert "Lagrange" in trace_names
    assert "PCHIP" in trace_names


def test_plot_yield_curve_returns_figure():
    fig = plot_yield_curve(X, Y)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_loo_errors_has_one_bar_per_method():
    results = validate_all_methods(X, Y)
    fig = plot_loo_errors(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == len(results)
    assert fig.layout.yaxis.type == "log"


def test_plot_pca_loadings_has_one_trace_per_component():
    loadings = pd.DataFrame(
        {"PC1": [0.3, 0.4, 0.3], "PC2": [0.5, 0.0, -0.5]}, index=[1.0, 5.0, 10.0]
    )
    fig = plot_pca_loadings(loadings)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_pca_scores_has_one_trace_per_component():
    dates = pd.date_range("2020-01-01", periods=5)
    scores = pd.DataFrame(
        {"PC1": [0.1, -0.2, 0.3, 0.0, -0.1], "PC2": [0.0, 0.1, -0.1, 0.2, 0.0]}, index=dates
    )
    fig = plot_pca_scores(scores)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_anomaly_scores_flags_the_right_points():
    dates = pd.date_range("2020-01-01", periods=5)
    scores = pd.Series([0.1, -0.9, 0.2, 0.15, -0.8], index=dates)
    is_anomaly = pd.Series([False, True, False, False, True], index=dates)
    fig = plot_anomaly_scores(scores, is_anomaly)
    assert isinstance(fig, go.Figure)
    flagged_trace = fig.data[1]
    assert len(flagged_trace.x) == 2
