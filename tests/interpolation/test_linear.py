import numpy as np
import pytest

from interpolation.linear import linear_interpolate


def test_exact_recovery_at_observed_nodes():
    x, y = [1.0, 5.0, 10.0], [4.15, 4.57, 4.80]
    np.testing.assert_allclose(linear_interpolate(x, y, x), y)


def test_midpoint_is_the_average_of_neighbors():
    # The defining property of a straight-line segment.
    estimate = linear_interpolate([0.0, 2.0], [4.0, 6.0], 1.0)
    assert estimate == pytest.approx(5.0)


def test_clamps_rather_than_extends_beyond_range():
    # Documented behavior: numpy.interp clamps to the boundary value.
    x, y = [1.0, 10.0], [4.0, 5.0]
    with pytest.warns(UserWarning, match="extrapolation"):
        estimate = linear_interpolate(x, y, 20.0)
    assert estimate == pytest.approx(5.0)  # clamped to the value at x=10, not extended


def test_warns_on_extrapolation():
    with pytest.warns(UserWarning, match="extrapolation"):
        linear_interpolate([1.0, 10.0], [4.0, 5.0], 0.5)
