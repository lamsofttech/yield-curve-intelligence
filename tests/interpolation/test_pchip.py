import numpy as np
import pytest

from interpolation.pchip import pchip_interpolate


def test_exact_recovery_at_observed_nodes():
    x, y = [1.0, 2.0, 5.0, 10.0], [4.15, 4.39, 4.57, 4.80]
    np.testing.assert_allclose(pchip_interpolate(x, y, x), y)


def test_warns_on_extrapolation():
    with pytest.warns(UserWarning, match="extrapolation"):
        pchip_interpolate([1.0, 2.0, 5.0, 10.0], [4.15, 4.39, 4.57, 4.80], 15.0)


def test_does_not_overshoot_on_the_same_step_like_dataset():
    # Same dataset that makes cubic spline overshoot (test_spline.py) --
    # PCHIP's defining guarantee is that it stays within the data's range.
    x, y = [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0]
    grid = np.linspace(0, 3, 200)
    values = pchip_interpolate(x, y, grid)
    assert values.min() >= 0.0
    assert values.max() <= 1.0
