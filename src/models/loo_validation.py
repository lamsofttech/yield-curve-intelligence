"""Leave-one-maturity-out validation of interpolation methods.

For each *interior* maturity of an observed curve (excludes the shortest
and longest -- removing either would turn the held-out estimate into
extrapolation, not interpolation, which is not what this measures), each
method in interpolation.METHODS is refit on the remaining nodes and asked
to estimate the omitted point; the estimate is compared against the real
observed value.

Limitations, stated up front (surfaced again in the app's Model Validation
section): a single curve typically has only ~9-11 interior maturities, so
MAE/RMSE/max-error here are small-sample estimates, not statistically
strong evidence that one method is "best" in general. They describe this
curve's shape on this date, not a guarantee about tomorrow's curve.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from interpolation import METHODS


@dataclass(frozen=True)
class LOOResult:
    """One method's leave-one-out validation result for one curve."""

    method: str
    errors_by_maturity: dict[float, float]  # maturity_years -> signed error (estimate - actual)
    mae: float
    rmse: float
    max_abs_error: float


def eligible_maturities(x_nodes: NDArray) -> NDArray:
    """Interior maturities only (excludes index 0 and -1). Exposed
    separately so the UI can show the user which maturities are excluded
    from validation, and why.
    """
    if len(x_nodes) < 3:
        return np.array([])
    return x_nodes[1:-1]


def leave_one_out_errors(x_nodes: NDArray, y_nodes: NDArray, method_fn) -> dict[float, float]:
    """Run leave-one-out for a single interpolation method.

    Returns {maturity_years: signed_error}, signed_error = estimate - actual.
    """
    errors: dict[float, float] = {}
    n = len(x_nodes)
    for i in range(1, n - 1):  # interior nodes only -- see module docstring
        x_reduced = np.delete(x_nodes, i)
        y_reduced = np.delete(y_nodes, i)
        with warnings.catch_warnings():
            # Interior nodes never require extrapolation by construction;
            # silencing here only suppresses a warning we already know
            # will not fire for a real reason.
            warnings.simplefilter("ignore", UserWarning)
            estimate = method_fn(x_reduced, y_reduced, float(x_nodes[i]))
        errors[float(x_nodes[i])] = float(estimate) - float(y_nodes[i])
    return errors


def validate_all_methods(x_nodes: NDArray, y_nodes: NDArray) -> list[LOOResult]:
    """Run leave-one-out validation for every registered method."""
    results = []
    for name, method_fn in METHODS.items():
        errors = leave_one_out_errors(x_nodes, y_nodes, method_fn)
        abs_errors = np.abs(np.array(list(errors.values())))
        results.append(
            LOOResult(
                method=name,
                errors_by_maturity=errors,
                mae=float(abs_errors.mean()) if len(abs_errors) else float("nan"),
                rmse=float(np.sqrt((abs_errors**2).mean())) if len(abs_errors) else float("nan"),
                max_abs_error=float(abs_errors.max()) if len(abs_errors) else float("nan"),
            )
        )
    return results


def rank_methods(results: list[LOOResult], by: str = "rmse") -> list[LOOResult]:
    """Sort results best-to-worst by the given metric ('mae', 'rmse', or
    'max_abs_error'). Purely mechanical -- see module docstring for why
    this ranking should be read cautiously, not as a general verdict.
    """
    return sorted(results, key=lambda r: getattr(r, by))
