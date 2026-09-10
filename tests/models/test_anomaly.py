import numpy as np
import pandas as pd
import pytest

from models.anomaly import current_percentile, fit_anomaly_detector, most_unusual_dates


def _synthetic_curve_levels(n_days: int = 300, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    maturities = [1 / 12, 1.0, 2.0, 5.0, 10.0, 30.0]
    base_curve = np.array([3.5, 3.8, 4.0, 4.3, 4.5, 4.6])
    normal_days = base_curve + rng.normal(0, 0.1, size=(n_days, len(maturities)))
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    panel = pd.DataFrame(normal_days, index=dates, columns=maturities)

    # Inject a handful of deliberately extreme/inverted days.
    extreme_dates = panel.index[[10, 50, 150]]
    panel.loc[extreme_dates] = [8.0, 2.0, 1.5, 1.0, 0.8, 0.5]  # sharply inverted, way off-distribution
    return panel


def test_fit_anomaly_detector_returns_score_and_flag_per_date():
    panel = _synthetic_curve_levels()
    result = fit_anomaly_detector(panel, contamination=0.05)
    assert len(result.scores) == len(panel)
    assert len(result.is_anomaly) == len(panel)
    assert result.is_anomaly.dtype == bool


def test_injected_extreme_days_are_flagged_as_anomalies():
    panel = _synthetic_curve_levels()
    result = fit_anomaly_detector(panel, contamination=0.05)
    extreme_dates = panel.index[[10, 50, 150]]
    # Not guaranteed for every possible random seed/dataset in general, but
    # for these deliberately extreme, far-off-distribution days it should
    # hold -- and did, verified by running this test.
    assert result.is_anomaly.loc[extreme_dates].all()


def test_most_unusual_dates_returns_requested_count_sorted_ascending_score():
    panel = _synthetic_curve_levels()
    result = fit_anomaly_detector(panel)
    top = most_unusual_dates(result, n=5)
    assert len(top) == 5
    assert top["anomaly_score"].is_monotonic_increasing  # lower score = more unusual, listed first


def test_current_percentile_is_between_0_and_100():
    panel = _synthetic_curve_levels()
    result = fit_anomaly_detector(panel)
    pct = current_percentile(result, panel.index[-1])
    assert 0 <= pct <= 100


def test_current_percentile_low_for_an_extreme_injected_day():
    panel = _synthetic_curve_levels()
    result = fit_anomaly_detector(panel)
    pct = current_percentile(result, panel.index[10])  # one of the injected extreme days
    assert pct < 20  # should rank near the "most unusual" end


def test_current_percentile_rejects_unknown_date():
    panel = _synthetic_curve_levels()
    result = fit_anomaly_detector(panel)
    with pytest.raises(ValueError):
        current_percentile(result, pd.Timestamp("1999-01-01"))
