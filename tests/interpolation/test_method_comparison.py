"""Tests using the full common METHODS interface, and the key empirical
result motivating this whole project: on the real, full Treasury maturity
grid, a global Lagrange polynomial catastrophically diverges in the sparse
10Y-20Y gap while the local methods (linear/spline/PCHIP) stay sane.
Verified numerically before writing (see LEARNING_LOG.md, Phase 4) --
Lagrange landed near -19,627% at 15Y, not a typo.
"""

import numpy as np

from interpolation import METHODS

# Real Treasury par yields, 2026-09-08, the full published grid.
FULL_X = np.array([0.0833, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
FULL_Y = np.array([3.81, 3.94, 4.00, 4.15, 4.39, 4.44, 4.57, 4.68, 4.80, 5.26, 5.25])


def test_registry_contains_all_four_methods():
    assert set(METHODS) == {"Lagrange", "Linear", "Cubic Spline", "PCHIP"}


def test_all_methods_exactly_recover_observed_nodes():
    for name, method in METHODS.items():
        np.testing.assert_allclose(method(FULL_X, FULL_Y, FULL_X), FULL_Y, atol=1e-6, err_msg=name)


def test_local_methods_agree_closely_in_the_sparse_gap():
    # Linear, cubic spline, and PCHIP all only look at local neighbors, so
    # they should broadly agree with each other in the 10Y-20Y gap even
    # though nothing anchors the shape there.
    target = 15.0
    local_estimates = [
        METHODS["Linear"](FULL_X, FULL_Y, target),
        METHODS["Cubic Spline"](FULL_X, FULL_Y, target),
        METHODS["PCHIP"](FULL_X, FULL_Y, target),
    ]
    assert max(local_estimates) - min(local_estimates) < 0.5  # within half a percentage point


def test_lagrange_diverges_wildly_from_local_methods_in_the_sparse_gap():
    # The core empirical result motivating this project's method-comparison
    # feature: a single global polynomial through all 11 real nodes
    # produces a nonsensical estimate in the wide gap, while local methods
    # do not. This is not a bug in our Lagrange implementation -- it is
    # what a global polynomial genuinely does with unevenly spaced nodes
    # (confirmed independently against scipy.BarycentricInterpolator in
    # test_lagrange.py).
    target = 15.0  # within [0.0833, 30.0] -- genuine interpolation, not extrapolation
    lagrange_estimate = METHODS["Lagrange"](FULL_X, FULL_Y, target)
    linear_estimate = METHODS["Linear"](FULL_X, FULL_Y, target)
    # A sane yield estimate is within roughly [0, 20]%; Lagrange's estimate
    # here is wildly outside that band, unlike every local method.
    assert abs(lagrange_estimate - linear_estimate) > 100
