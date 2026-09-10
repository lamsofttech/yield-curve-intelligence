import numpy as np
import pytest

from interpolation.spline import cubic_spline_interpolate


def test_exact_recovery_at_observed_nodes():
    x, y = [1.0, 2.0, 5.0, 10.0], [4.15, 4.39, 4.57, 4.80]
    np.testing.assert_allclose(cubic_spline_interpolate(x, y, x), y)


def test_warns_on_extrapolation():
    with pytest.warns(UserWarning, match="extrapolation"):
        cubic_spline_interpolate([1.0, 2.0, 5.0, 10.0], [4.15, 4.39, 4.57, 4.80], 15.0)


def test_overshoots_on_a_step_like_dataset():
    # The property that motivates PCHIP existing at all: natural cubic
    # spline is not shape-preserving. Verified numerically before writing
    # this test -- see LEARNING_LOG.md, Phase 4.
    x, y = [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0]
    grid = np.linspace(0, 3, 200)
    values = cubic_spline_interpolate(x, y, grid)
    assert values.min() < 0.0 or values.max() > 1.0
