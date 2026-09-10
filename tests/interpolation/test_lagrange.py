import numpy as np
import pytest
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import lagrange as scipy_lagrange

from interpolation.lagrange import lagrange_basis, lagrange_interpolate

# Real Treasury par yields, 2026-09-08 (see DECISIONS.md).
REAL_X = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
REAL_Y = np.array([4.15, 4.39, 4.44, 4.57, 4.68, 4.80])


def test_exact_recovery_at_observed_nodes():
    # The defining property: P(x_j) must equal y_j exactly for every node.
    estimate = lagrange_interpolate(REAL_X, REAL_Y, REAL_X)
    np.testing.assert_allclose(estimate, REAL_Y)


def test_worked_example_matches_hand_derivation():
    # From the Phase 3 write-up: 1Yr/5Yr/10Yr nodes only, estimate at 3Yr.
    x = np.array([1.0, 5.0, 10.0])
    y = np.array([4.15, 4.57, 4.80])
    estimate = lagrange_interpolate(x, y, 3.0)
    assert estimate == pytest.approx(4.386, abs=1e-3)


def test_basis_polynomials_sum_to_one():
    # Partition-of-unity property: sum_j L_j(x) == 1 for any x (proof: it's
    # the unique degree-(n-1) polynomial through n points all equal to 1).
    total = sum(lagrange_basis(REAL_X, j, 4.0) for j in range(len(REAL_X)))
    assert total == pytest.approx(1.0)


def test_recovers_a_known_quadratic_between_nodes():
    x = np.array([-2.0, 0.0, 1.0, 3.0])
    y = 3 * x**2 - 2 * x + 4
    grid = np.linspace(-2.0, 3.0, 25)
    expected = 3 * grid**2 - 2 * grid + 4
    np.testing.assert_allclose(lagrange_interpolate(x, y, grid), expected, atol=1e-8)


def test_scalar_input_returns_python_float():
    value = lagrange_interpolate([1.0, 2.0], [4.0, 5.0], 1.5)
    assert isinstance(value, float)


def test_warns_on_extrapolation_beyond_observed_range():
    with pytest.warns(UserWarning, match="extrapolation"):
        lagrange_interpolate(REAL_X, REAL_Y, 15.0)  # beyond the 10.0 max node


def test_no_warning_within_observed_range(warnings_as_errors):
    with warnings_as_errors:
        lagrange_interpolate(REAL_X, REAL_Y, 4.0)


@pytest.mark.parametrize(
    ("x", "y", "match"),
    [
        ([], [], "non-zero"),
        ([1.0, 1.0], [4.0, 5.0], "distinct"),
        ([1.0], [4.0, 5.0], "same non-zero"),
        ([1.0, float("nan")], [4.0, 5.0], "finite"),
    ],
)
def test_rejects_invalid_nodes(x, y, match):
    with pytest.raises(ValueError, match=match):
        lagrange_interpolate(x, y, 1.5)


def test_matches_scipy_barycentric_interpolator():
    # Cross-check our from-scratch implementation against a trusted library,
    # on the real yield-curve nodes, across the full observed maturity range.
    scipy_interp = BarycentricInterpolator(REAL_X, REAL_Y)
    grid = np.linspace(REAL_X.min(), REAL_X.max(), 50)
    ours = lagrange_interpolate(REAL_X, REAL_Y, grid)
    theirs = scipy_interp(grid)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_matches_scipy_lagrange_function_on_small_node_set():
    # scipy.interpolate.lagrange builds the same mathematical polynomial via
    # numpy.poly1d, which scipy's own docs warn is numerically unstable for
    # larger node counts -- fine to cross-check on a small set like this,
    # but BarycentricInterpolator above is the more robust comparison and
    # the reason production code doesn't use scipy.interpolate.lagrange.
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.15, 4.39, 4.44])
    poly = scipy_lagrange(x, y)
    grid = np.linspace(1.0, 3.0, 10)
    np.testing.assert_allclose(lagrange_interpolate(x, y, grid), poly(grid), atol=1e-8)
