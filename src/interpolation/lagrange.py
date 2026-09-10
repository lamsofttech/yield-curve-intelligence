"""Lagrange polynomial interpolation, implemented from the defining formula.

Deliberately not calling scipy.interpolate.lagrange or numpy.polynomial:
the point of this module is that the mechanism is visible, not hidden
behind a library call. This is the classical/direct form (evaluate each
basis polynomial by its product formula), which is O(n^2) per evaluation
point -- clear and directly traceable to the math, though not the most
numerically efficient form (production libraries use a "barycentric"
reformulation; see DECISIONS.md).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from interpolation.base import as_scalar_or_array, validate_nodes, warn_if_extrapolating


def lagrange_basis(x_nodes: ArrayLike, j: int, x_eval: float) -> float:
    """Evaluate the j-th Lagrange basis polynomial L_j at x_eval.

    L_j(x) = product over k != j of (x - x_k) / (x_j - x_k)

    L_j equals 1 at x_nodes[j] and 0 at every other node -- the "switch"
    property that makes the sum in lagrange_interpolate pass exactly
    through every observed point.
    """
    x = np.asarray(x_nodes, dtype=float)
    xj = x[j]
    others = np.delete(x, j)
    return float(np.prod((x_eval - others) / (xj - others)))


def lagrange_interpolate(x_nodes: ArrayLike, y_nodes: ArrayLike, x_eval: ArrayLike):
    """Evaluate the unique degree-(n-1) polynomial through n given nodes.

    P(x) = sum_j y_j * L_j(x)
    """
    x, y = validate_nodes(x_nodes, y_nodes)
    points = np.atleast_1d(np.asarray(x_eval, dtype=float))
    if not np.isfinite(points).all():
        raise ValueError("x_eval must contain only finite values")
    warn_if_extrapolating(x, points, "lagrange_interpolate")

    n = len(x)
    result = np.empty_like(points)
    for i, xe in enumerate(points):
        basis_values = np.array([lagrange_basis(x, j, xe) for j in range(n)])
        result[i] = float(np.dot(y, basis_values))

    return as_scalar_or_array(result, x_eval)
