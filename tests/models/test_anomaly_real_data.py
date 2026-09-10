"""Anomaly detection sanity checks against the real bundled 2016-2026
historical data. Verified numerically before writing (see LEARNING_LOG.md,
Phase 6): the model independently flags October 2023 (the real Treasury
yield spike to ~5%, highest since 2007) and March 2020 (COVID market
panic) as among the most unusual dates in the whole window -- both
genuine, well-known market events, not arbitrary.
"""

import pandas as pd

from data.loader import load_historical
from features.yield_changes import to_wide_panel
from models.anomaly import current_percentile, fit_anomaly_detector, most_unusual_dates


def _real_anomaly_result():
    tidy = load_historical()
    panel = to_wide_panel(tidy)
    return fit_anomaly_detector(panel), panel


def test_october_2023_yield_spike_is_flagged_anomalous():
    result, _ = _real_anomaly_result()
    # The real Treasury 10Y yield spike to ~5% (highest since 2007).
    assert result.is_anomaly.loc[pd.Timestamp("2023-10-19")]


def test_covid_crash_onset_is_flagged_anomalous():
    result, _ = _real_anomaly_result()
    assert result.is_anomaly.loc[pd.Timestamp("2020-03-06")]


def test_most_unusual_dates_are_dominated_by_known_stress_periods():
    result, _ = _real_anomaly_result()
    top_20 = most_unusual_dates(result, n=20)
    # At least half of the 20 most unusual dates fall in known stress
    # windows (Oct 2023 yield spike, COVID onset Feb-Mar 2020).
    in_known_window = top_20.index.to_series().apply(
        lambda d: (pd.Timestamp("2023-10-01") <= d <= pd.Timestamp("2023-11-05"))
        or (pd.Timestamp("2020-02-15") <= d <= pd.Timestamp("2020-03-31"))
    )
    assert in_known_window.sum() >= 10


def test_current_percentile_is_a_valid_percentile_on_real_data():
    result, panel = _real_anomaly_result()
    pct = current_percentile(result, panel.index.max())
    assert 0 <= pct <= 100
