import pandas as pd
import pytest

from features.yield_changes import COMMON_MATURITIES, daily_changes, to_wide_panel


def _tidy_two_days() -> pd.DataFrame:
    rows = []
    for date, values in [
        ("2026-01-02", [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.5, 4.6]),
        ("2026-01-03", [3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.6, 4.7]),
    ]:
        for maturity, value in zip(COMMON_MATURITIES, values, strict=True):
            rows.append({"date": pd.Timestamp(date), "maturity_years": maturity, "yield_pct": value})
    return pd.DataFrame(rows)


def test_to_wide_panel_has_one_row_per_date_and_one_column_per_maturity():
    wide = to_wide_panel(_tidy_two_days())
    assert wide.shape == (2, len(COMMON_MATURITIES))
    assert list(wide.columns) == COMMON_MATURITIES


def test_to_wide_panel_drops_dates_missing_a_required_maturity():
    tidy = _tidy_two_days()
    # Remove the 30Y observation for the second date only.
    tidy = tidy[~((tidy["date"] == pd.Timestamp("2026-01-03")) & (tidy["maturity_years"] == 30.0))]
    wide = to_wide_panel(tidy)
    assert len(wide) == 1
    assert wide.index[0] == pd.Timestamp("2026-01-02")


def test_to_wide_panel_rejects_unavailable_maturity():
    with pytest.raises(ValueError, match="not present"):
        to_wide_panel(_tidy_two_days(), maturities=[99.0])


def test_daily_changes_drops_first_row_and_computes_diff():
    wide = to_wide_panel(_tidy_two_days())
    changes = daily_changes(wide)
    assert len(changes) == 1  # one day of data lost to differencing
    assert changes.iloc[0].tolist() == pytest.approx([0.1] * len(COMMON_MATURITIES))
