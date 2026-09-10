import numpy as np
import pytest

from interpolation.base import as_scalar_or_array, validate_nodes, warn_if_extrapolating


def test_validate_nodes_accepts_valid_input():
    x, y = validate_nodes([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("x", "y", "match"),
    [
        ([], [], "non-zero"),
        ([1.0, 1.0], [4.0, 5.0], "distinct"),
        ([1.0], [4.0, 5.0], "same non-zero"),
        ([1.0, float("nan")], [4.0, 5.0], "finite"),
    ],
)
def test_validate_nodes_rejects_invalid_input(x, y, match):
    with pytest.raises(ValueError, match=match):
        validate_nodes(x, y)


def test_warn_if_extrapolating_fires_outside_range():
    x = np.array([1.0, 5.0, 10.0])
    with pytest.warns(UserWarning, match="extrapolation"):
        warn_if_extrapolating(x, np.array([15.0]), "test_method")


def test_warn_if_extrapolating_silent_inside_range(warnings_as_errors):
    x = np.array([1.0, 5.0, 10.0])
    with warnings_as_errors:
        warn_if_extrapolating(x, np.array([4.0]), "test_method")  # must not raise/warn


def test_as_scalar_or_array_matches_input_shape():
    assert isinstance(as_scalar_or_array(np.array([4.2]), 1.5), float)
    assert isinstance(as_scalar_or_array(np.array([4.2, 5.1]), [1.5, 2.5]), np.ndarray)
