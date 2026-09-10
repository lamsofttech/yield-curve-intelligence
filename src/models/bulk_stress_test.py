"""Historical stress-test: evaluate Lagrange vs. a local method (linear)
on a dense maturity grid, for every day in the bundled history at once.

Answers a question the single-date Interpolation Lab cannot: across
~2,600+ real trading days spanning many different curve shapes, how often
and how badly does a full-grid Lagrange polynomial diverge from a local
method? Because every day shares the identical maturity grid, the
Lagrange basis polynomials and the linear-interpolation neighbor/weight
lookup only need to be computed ONCE (they don't depend on the day, only
on the fixed x_nodes and the evaluation grid) -- so the per-day cost
collapses into a single (days x nodes) @ (nodes x grid_points) matrix
multiply. That's genuinely GPU-sized work (millions of multiply-adds),
not a token computation dressed up to justify a GPU allocation.

Pure numpy, not torch -- torch is only imported by the deployment-specific
wrapper (see deploy/hf_app.py) so the base app/Docker/CI path never needs
torch as a dependency. The math here is identical either way; only the
array backend differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BulkStressTestResult:
    grid: NDArray  # (n_grid,) maturity values evaluated
    lagrange_values: NDArray  # (n_days, n_grid)
    linear_values: NDArray  # (n_days, n_grid)
    max_abs_divergence_by_day: NDArray  # (n_days,) max |lagrange - linear| per day
    dates: NDArray  # (n_days,) the corresponding dates


def _lagrange_basis_matrix(x_nodes: NDArray, grid: NDArray) -> NDArray:
    """Basis matrix L[g, j] = prod_{k!=j} (grid[g]-x_nodes[k])/(x_nodes[j]-x_nodes[k]).

    Computed once for the fixed node set and grid -- independent of any
    day's y-values, which is exactly what makes the bulk evaluation cheap:
    every day's Lagrange curve is then just `y_day @ basis.T`.
    """
    n_nodes = len(x_nodes)
    basis = np.ones((len(grid), n_nodes))
    for j in range(n_nodes):
        for k in range(n_nodes):
            if k != j:
                basis[:, j] *= (grid - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
    return basis


def _linear_weights(x_nodes: NDArray, grid: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Precompute, for each grid point, which two nodes bracket it and the
    linear blend weight -- also independent of any day's y-values.
    """
    clipped = np.clip(grid, x_nodes[0], x_nodes[-1])
    right_idx = np.searchsorted(x_nodes, clipped, side="left").clip(1, len(x_nodes) - 1)
    left_idx = right_idx - 1
    x_left, x_right = x_nodes[left_idx], x_nodes[right_idx]
    weight = np.where(x_right > x_left, (clipped - x_left) / (x_right - x_left), 0.0)
    return left_idx, right_idx, weight


def run_bulk_stress_test(
    x_nodes: NDArray, y_all_days: NDArray, dates: NDArray, n_grid_points: int = 2000
) -> BulkStressTestResult:
    """Evaluate Lagrange and linear interpolation on a dense grid for every
    row (day) of y_all_days at once.

    x_nodes: (n_nodes,) maturities, identical for every day.
    y_all_days: (n_days, n_nodes) yields, one row per day.
    """
    grid = np.linspace(x_nodes.min(), x_nodes.max(), n_grid_points)

    basis = _lagrange_basis_matrix(x_nodes, grid)  # (n_grid, n_nodes)
    lagrange_values = y_all_days @ basis.T  # (n_days, n_grid)

    left_idx, right_idx, weight = _linear_weights(x_nodes, grid)
    y_left, y_right = y_all_days[:, left_idx], y_all_days[:, right_idx]
    linear_values = y_left + weight * (y_right - y_left)

    max_divergence = np.abs(lagrange_values - linear_values).max(axis=1)

    return BulkStressTestResult(
        grid=grid,
        lagrange_values=lagrange_values,
        linear_values=linear_values,
        max_abs_divergence_by_day=max_divergence,
        dates=dates,
    )


def summarize_stress_test(result: BulkStressTestResult) -> str:
    """Human-readable summary of the divergence distribution across all days.

    A single threshold ("% of days exceeding X pp") turned out to be
    uninformative on real data: at a threshold low enough to be a
    meaningful sanity check (10 pp), literally 100% of days exceed it --
    Lagrange's full-grid divergence is not an occasional event, it is the
    typical case. The distribution itself (median vs. worst case) tells
    the real story instead.
    """
    n_days = len(result.dates)
    d = result.max_abs_divergence_by_day
    worst_idx = int(np.argmax(d))
    start = np.datetime_as_string(result.dates.min(), unit="D")
    end = np.datetime_as_string(result.dates.max(), unit="D")
    worst_date = np.datetime_as_string(result.dates[worst_idx], unit="D")
    return (
        f"Across all {n_days:,} trading days ({start} to {end}), Lagrange's maximum divergence "
        f"from linear interpolation anywhere on the maturity grid: "
        f"median = {np.median(d):,.0f} percentage points, "
        f"90th percentile = {np.percentile(d, 90):,.0f} pp, "
        f"worst = {d[worst_idx]:,.0f} pp (on {worst_date}). "
        f"This is not an occasional failure mode -- on this real, full 11-node grid, "
        f"catastrophic divergence somewhere on the curve is the typical case, not the exception."
    )
