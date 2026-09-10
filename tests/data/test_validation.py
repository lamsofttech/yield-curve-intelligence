import pandas as pd
import pytest

from data.validation import (
    validate_curve_data,
    validate_maturity_ordering,
    validate_no_duplicate_observations,
    validate_required_columns,
    validate_yield_range,
)


def _valid_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-08", "2026-09-08", "2026-09-08"]),
            "maturity_years": [0.25, 1.0, 10.0],
            "maturity_label": ["3 Mo", "1 Yr", "10 Yr"],
            "yield_pct": [3.94, 4.15, 4.80],
        }
    )


def test_validate_required_columns_accepts_valid_table():
    validate_required_columns(_valid_table())  # must not raise


def test_validate_required_columns_rejects_missing_column():
    bad = _valid_table().drop(columns=["yield_pct"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(bad)


def test_validate_required_columns_rejects_wrong_dtype():
    bad = _valid_table()
    bad["yield_pct"] = bad["yield_pct"].astype(str)
    with pytest.raises(ValueError, match="must be float"):
        validate_required_columns(bad)


def test_validate_no_duplicates_accepts_valid_table():
    validate_no_duplicate_observations(_valid_table())  # must not raise


def test_validate_no_duplicates_rejects_duplicate_observation():
    bad = pd.concat([_valid_table(), _valid_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_no_duplicate_observations(bad)


def test_validate_yield_range_accepts_valid_table():
    validate_yield_range(_valid_table())  # must not raise


def test_validate_yield_range_rejects_implausible_yield():
    bad = _valid_table()
    bad.loc[0, "yield_pct"] = 415.0  # e.g. a units error (4.15 -> 415)
    with pytest.raises(ValueError, match="outside"):
        validate_yield_range(bad)


def test_validate_maturity_ordering_accepts_sorted_table():
    validate_maturity_ordering(_valid_table())  # already sorted: 0.25, 1.0, 10.0


def test_validate_maturity_ordering_rejects_out_of_order_table():
    bad = _valid_table().iloc[[1, 0, 2]].reset_index(drop=True)  # 1.0, 0.25, 10.0 -- out of order
    with pytest.raises(ValueError):
        validate_maturity_ordering(bad)


def test_validate_maturity_ordering_rejects_equal_neighbors():
    # Series.is_monotonic_increasing would wrongly accept this (it means
    # "non-decreasing"); this table has two rows at maturity_years=1.0 for
    # the same date, which must be rejected as not *strictly* increasing.
    bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-08"] * 3),
            "maturity_years": [0.25, 1.0, 1.0],
            "maturity_label": ["3 Mo", "1 Yr", "1 Yr (dup)"],
            "yield_pct": [3.94, 4.15, 4.15],
        }
    )
    with pytest.raises(ValueError):
        validate_maturity_ordering(bad)


def test_validate_curve_data_runs_all_checks():
    validate_curve_data(_valid_table())  # must not raise
