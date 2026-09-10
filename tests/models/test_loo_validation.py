import numpy as np
import pytest

from models.loo_validation import (
    eligible_maturities,
    leave_one_out_errors,
    rank_methods,
    validate_all_methods,
)

# Real Treasury par yields, 2026-09-08, full published grid.
FULL_X = np.array([0.0833, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
FULL_Y = np.array([3.81, 3.94, 4.00, 4.15, 4.39, 4.44, 4.57, 4.68, 4.80, 5.26, 5.25])


def test_eligible_maturities_excludes_endpoints():
    eligible = eligible_maturities(FULL_X)
    assert eligible[0] == FULL_X[1]
    assert eligible[-1] == FULL_X[-2]
    assert len(eligible) == len(FULL_X) - 2


def test_eligible_maturities_empty_for_too_few_nodes():
    assert len(eligible_maturities(np.array([1.0, 2.0]))) == 0


def test_leave_one_out_errors_covers_only_interior_nodes():
    from interpolation import METHODS

    errors = leave_one_out_errors(FULL_X, FULL_Y, METHODS["Linear"])
    assert set(errors.keys()) == set(eligible_maturities(FULL_X).tolist())


def test_leave_one_out_error_is_zero_for_a_perfectly_linear_curve():
    # If the underlying curve genuinely is a straight line, linear
    # interpolation must recover each held-out point exactly.
    from interpolation import METHODS

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 4.0 + 0.1 * x  # exactly linear
    errors = leave_one_out_errors(x, y, METHODS["Linear"])
    for err in errors.values():
        assert err == pytest.approx(0.0, abs=1e-10)


def test_validate_all_methods_returns_one_result_per_method():
    results = validate_all_methods(FULL_X, FULL_Y)
    assert {r.method for r in results} == {"Lagrange", "Linear", "Cubic Spline", "PCHIP"}
    for r in results:
        assert r.mae >= 0
        assert r.rmse >= 0
        assert r.max_abs_error >= 0


def test_local_methods_vastly_outperform_lagrange_on_full_real_grid():
    # The empirical result motivating this whole phase: on the real, full
    # 11-node curve, Lagrange's leave-one-out errors are catastrophically
    # large (millions of percentage points) while local methods are all
    # well under 1 percentage point. Verified numerically before writing
    # this test (see LEARNING_LOG.md, Phase 5).
    results = {r.method: r for r in validate_all_methods(FULL_X, FULL_Y)}
    for local_method in ("Linear", "Cubic Spline", "PCHIP"):
        assert results[local_method].rmse < 1.0
    assert results["Lagrange"].rmse > 1000


def test_rank_methods_orders_best_first():
    results = validate_all_methods(FULL_X, FULL_Y)
    ranked = rank_methods(results, by="rmse")
    rmses = [r.rmse for r in ranked]
    assert rmses == sorted(rmses)
