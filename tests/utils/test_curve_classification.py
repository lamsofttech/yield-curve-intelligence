import pytest

from utils.curve_classification import classify_curve, compute_spreads


def test_classify_normal():
    assert classify_curve(spread_10y_2y=0.5, spread_10y_3m=0.8) == "Normal"


def test_classify_inverted():
    assert classify_curve(spread_10y_2y=-0.5, spread_10y_3m=-0.8) == "Inverted"


def test_classify_flat():
    assert classify_curve(spread_10y_2y=0.02, spread_10y_3m=-0.03) == "Flat"


def test_classify_mixed_when_spreads_disagree():
    assert classify_curve(spread_10y_2y=0.3, spread_10y_3m=-0.3) == "Mixed"


def test_real_current_curve_classifies_as_normal():
    # 2026-09-08 real Treasury data.
    curve = {10.0: 4.80, 2.0: 4.39, 0.25: 3.94}
    spreads = compute_spreads(curve)
    assert classify_curve(*spreads) == "Normal"


def test_real_2023_curve_classifies_as_inverted():
    # 2023-10-19 real Treasury data -- the well-known inversion era.
    curve = {10.0: 4.98, 2.0: 5.14, 0.25: 5.60}
    spreads = compute_spreads(curve)
    assert classify_curve(*spreads) == "Inverted"


def test_compute_spreads_raises_on_missing_maturity():
    with pytest.raises(KeyError, match="missing maturities"):
        compute_spreads({10.0: 4.80, 2.0: 4.39})  # no 3-month point
