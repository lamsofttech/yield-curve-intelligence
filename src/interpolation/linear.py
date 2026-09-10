"""Piecewise-linear interpolation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from interpolation.base import as_scalar_or_array, validate_nodes, warn_if_extrapolating


def linear_interpolate(x_nodes: ArrayLike, y_nodes: ArrayLike, x_eval: ArrayLike):
    """Connect each pair of adjacent observed points with a straight line.

    Tradeoffs: simple and robust -- it can never oscillate or overshoot,
    since each segment only depends on its two endpoints. But it has a
    "kink" (discontinuous slope) at every node, and it treats every gap
    between nodes as equally trustworthy regardless of width -- unlike
    Lagrange, a wide gap (e.g. 10Y-20Y) doesn't make linear interpolation
    less reliable there, but it also can't capture curvature within a gap.

    Note: numpy.interp does not actually extrapolate outside the node
    range -- it clamps to the nearest boundary value (flat beyond the
    endpoints), unlike Lagrange/spline/PCHIP below, which continue their
    curve. The extrapolation warning still fires either way, since a
    clamped value is just as unreliable a "prediction" as an extended one.
    """
    x, y = validate_nodes(x_nodes, y_nodes)
    points = np.atleast_1d(np.asarray(x_eval, dtype=float))
    warn_if_extrapolating(x, points, "linear_interpolate")
    result = np.interp(points, x, y)
    return as_scalar_or_array(result, x_eval)
