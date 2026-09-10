"""Natural cubic spline interpolation (wraps scipy.interpolate.CubicSpline).

Not hand-implemented: deriving cubic spline coefficients means solving a
tridiagonal system for the second-derivative continuity conditions, which
is a substantial derivation on its own. The pedagogical focus of this
project is Lagrange (built from scratch); the other methods are used as
well-understood, trusted library implementations so we can focus on
*comparing* their behavior and tradeoffs rather than re-deriving each one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from interpolation.base import as_scalar_or_array, validate_nodes, warn_if_extrapolating


def cubic_spline_interpolate(x_nodes: ArrayLike, y_nodes: ArrayLike, x_eval: ArrayLike):
    """Piecewise cubic polynomials, joined so the curve, its slope, and its
    curvature (first and second derivatives) are all continuous at every
    node.

    Tradeoffs: much smoother than linear, and far less prone to the wild
    oscillation a single high-degree global polynomial (Lagrange) can show
    in sparse regions -- each piece only depends on its local neighbors,
    typically the four nearest nodes, not all of them at once. Can still
    slightly overshoot near a sharp change in curve direction (unlike
    PCHIP, which explicitly forbids that).
    """
    x, y = validate_nodes(x_nodes, y_nodes)
    points = np.atleast_1d(np.asarray(x_eval, dtype=float))
    warn_if_extrapolating(x, points, "cubic_spline_interpolate")
    spline = CubicSpline(x, y, extrapolate=True)
    result = spline(points)
    return as_scalar_or_array(result, x_eval)
