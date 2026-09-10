import numpy as np

from interpolation import METHODS
from models.bulk_stress_test import run_bulk_stress_test, summarize_stress_test


def test_bulk_lagrange_matches_trusted_per_day_implementation():
    # Small synthetic panel: 3 days, 4 nodes, deliberately unevenly spaced
    # to actually exercise a nontrivial Lagrange shape.
    x_nodes = np.array([1.0, 2.0, 5.0, 10.0])
    y_all_days = np.array(
        [
            [4.0, 4.2, 4.5, 4.8],
            [3.5, 4.0, 4.6, 5.0],
            [4.9, 4.7, 4.3, 4.1],
        ]
    )
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")

    result = run_bulk_stress_test(x_nodes, y_all_days, dates, n_grid_points=50)

    for day_idx in range(3):
        for grid_idx in (0, 25, 49):
            target = float(result.grid[grid_idx])
            trusted = METHODS["Lagrange"](x_nodes, y_all_days[day_idx], target)
            actual = result.lagrange_values[day_idx, grid_idx]
            np.testing.assert_allclose(actual, trusted, atol=1e-8)


def test_bulk_linear_matches_trusted_per_day_implementation():
    x_nodes = np.array([1.0, 2.0, 5.0, 10.0])
    y_all_days = np.array([[4.0, 4.2, 4.5, 4.8], [3.5, 4.0, 4.6, 5.0]])
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")

    result = run_bulk_stress_test(x_nodes, y_all_days, dates, n_grid_points=50)

    for day_idx in range(2):
        for grid_idx in (0, 10, 49):
            target = float(result.grid[grid_idx])
            trusted = METHODS["Linear"](x_nodes, y_all_days[day_idx], target)
            np.testing.assert_allclose(result.linear_values[day_idx, grid_idx], trusted, atol=1e-8)


def test_grid_exactly_at_nodes_recovers_observed_values():
    x_nodes = np.array([1.0, 2.0, 5.0, 10.0])
    y_all_days = np.array([[4.0, 4.2, 4.5, 4.8]])
    dates = np.array(["2020-01-01"], dtype="datetime64[D]")

    # Use the nodes themselves as the grid.
    result = run_bulk_stress_test(x_nodes, y_all_days, dates, n_grid_points=4)
    # (linspace over the same range with 4 points recovers the 4 nodes exactly
    #  only because they happen to align here -- verify via direct recovery
    #  at the endpoints, which linspace always includes.)
    np.testing.assert_allclose(result.lagrange_values[0, 0], y_all_days[0, 0], atol=1e-8)
    np.testing.assert_allclose(result.lagrange_values[0, -1], y_all_days[0, -1], atol=1e-8)
    np.testing.assert_allclose(result.linear_values[0, 0], y_all_days[0, 0], atol=1e-8)
    np.testing.assert_allclose(result.linear_values[0, -1], y_all_days[0, -1], atol=1e-8)


def test_summarize_stress_test_produces_readable_string():
    x_nodes = np.array([1.0, 2.0, 5.0, 10.0])
    y_all_days = np.array([[4.0, 4.2, 4.5, 4.8], [3.5, 4.0, 4.6, 5.0]])
    dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    result = run_bulk_stress_test(x_nodes, y_all_days, dates, n_grid_points=50)

    summary = summarize_stress_test(result)
    assert "median" in summary
    assert "worst" in summary
    assert "2020-01-01" in summary or "2020-01-02" in summary
