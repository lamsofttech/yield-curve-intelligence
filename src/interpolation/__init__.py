"""Four interpolation methods behind one common interface:

    method(x_nodes, y_nodes, x_eval) -> float | ndarray

METHODS lets calling code (leave-one-out validation, the comparison chart)
loop over every method identically instead of an if/elif chain.
"""

from interpolation.lagrange import lagrange_basis, lagrange_interpolate
from interpolation.linear import linear_interpolate
from interpolation.pchip import pchip_interpolate
from interpolation.spline import cubic_spline_interpolate

METHODS = {
    "Lagrange": lagrange_interpolate,
    "Linear": linear_interpolate,
    "Cubic Spline": cubic_spline_interpolate,
    "PCHIP": pchip_interpolate,
}

__all__ = [
    "METHODS",
    "lagrange_basis",
    "lagrange_interpolate",
    "linear_interpolate",
    "cubic_spline_interpolate",
    "pchip_interpolate",
]
