"""Shared validation and extrapolation-warning logic used by every
interpolation method, plus the common interface they all implement:

    method(x_nodes, y_nodes, x_eval) -> float | ndarray

A shared signature (rather than, say, four differently-shaped classes)
keeps this simple: the leave-one-out validation in Phase 5 and the
comparison chart in Phase 7 can loop over every method identically.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_nodes(x_nodes: ArrayLike, y_nodes: ArrayLike) -> tuple[NDArray, NDArray]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be one-dimensional")
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("x_nodes and y_nodes must have the same non-zero length")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nodes must contain only finite values")
    if len(np.unique(x)) != len(x):
        raise ValueError("x_nodes must be distinct (duplicate maturities)")
    return x, y


def warn_if_extrapolating(x_nodes: NDArray, x_eval: NDArray, method_name: str) -> None:
    """Warn (not raise) when any evaluation point falls outside the observed
    maturity range -- per the project's "reject or warn on extrapolation"
    requirement. A polynomial or spline is still numerically well-defined
    outside the node range, but the estimate there is not something this
    project treats as trustworthy.
    """
    lo, hi = x_nodes.min(), x_nodes.max()
    if ((x_eval < lo) | (x_eval > hi)).any():
        warnings.warn(
            f"{method_name}: evaluating outside the observed maturity range [{lo}, {hi}] "
            "-- this is extrapolation, not interpolation, and is not reliable.",
            stacklevel=3,
        )


def as_scalar_or_array(result: NDArray, x_eval: ArrayLike):
    """Match the output shape convention: scalar in -> Python float out,
    array in -> ndarray out. Keeps every method's return type predictable
    for callers regardless of which one they invoked.
    """
    return float(result[0]) if np.ndim(x_eval) == 0 else np.asarray(result, dtype=float)
