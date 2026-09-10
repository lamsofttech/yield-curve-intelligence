"""Shape-preserving piecewise cubic Hermite interpolation (PCHIP).

Wraps scipy.interpolate.PchipInterpolator -- see spline.py's module
docstring for why this isn't hand-derived.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

from interpolation.base import as_scalar_or_array, validate_nodes, warn_if_extrapolating


def pchip_interpolate(x_nodes: ArrayLike, y_nodes: ArrayLike, x_eval: ArrayLike):
    """Piecewise cubic, like cubic_spline_interpolate, but constrained to be
    "shape-preserving": it will never introduce a local max or min between
    two nodes that isn't already implied by the data (no overshoot).

    Tradeoffs: the overshoot guarantee is valuable when a curve should be
    monotonic between certain points, but it costs some smoothness -- PCHIP
    is only once continuously differentiable (curvature can jump slightly
    at a node), where cubic spline is twice continuously differentiable.
    """
    x, y = validate_nodes(x_nodes, y_nodes)
    points = np.atleast_1d(np.asarray(x_eval, dtype=float))
    warn_if_extrapolating(x, points, "pchip_interpolate")
    interpolator = PchipInterpolator(x, y, extrapolate=True)
    result = interpolator(points)
    return as_scalar_or_array(result, x_eval)
