"""Validation checks for tidy Treasury yield curve data.

Every function here takes the tidy long-format table (date, maturity_years,
maturity_label, yield_pct) and either returns silently (valid) or raises
ValueError with a message specific enough to actually debug from.
"""

from __future__ import annotations

import pandas as pd

from data.treasury_client import TIDY_COLUMNS


def validate_required_columns(table: pd.DataFrame) -> None:
    """Check the table has exactly the expected columns, with sane dtypes."""
    missing = set(TIDY_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(table["date"]):
        raise ValueError(f"'date' column must be datetime64, got {table['date'].dtype}")
    if not pd.api.types.is_float_dtype(table["maturity_years"]):
        raise ValueError(f"'maturity_years' must be float, got {table['maturity_years'].dtype}")
    if not pd.api.types.is_float_dtype(table["yield_pct"]):
        raise ValueError(f"'yield_pct' must be float, got {table['yield_pct'].dtype}")


def validate_no_duplicate_observations(table: pd.DataFrame) -> None:
    """Check there is at most one yield observation per (date, maturity)."""
    duplicated = table.duplicated(subset=["date", "maturity_years"], keep=False)
    if duplicated.any():
        bad = table.loc[duplicated, ["date", "maturity_years"]].drop_duplicates()
        raise ValueError(f"Duplicate (date, maturity_years) observations found:\n{bad}")


def validate_yield_range(table: pd.DataFrame, min_pct: float = -2.0, max_pct: float = 25.0) -> None:
    """Sanity-check yields fall in a plausible range.

    Wide bounds on purpose -- this is a defense against corrupted/garbled
    data (e.g. a units error turning 4.15 into 415), not an economic model
    of what yields "should" be. -2% allows for the kind of mild negative
    short-term rates seen in some markets historically, even though U.S.
    Treasuries have not gone negative to date.
    """
    out_of_range = table[(table["yield_pct"] < min_pct) | (table["yield_pct"] > max_pct)]
    if not out_of_range.empty:
        raise ValueError(
            f"{len(out_of_range)} yield observation(s) outside [{min_pct}, {max_pct}]%:\n"
            f"{out_of_range[['date', 'maturity_label', 'yield_pct']]}"
        )


def validate_maturity_ordering(table: pd.DataFrame) -> None:
    """Check that, for every date, maturities appear in strictly increasing order.

    TODO(you): implement this one.

    Why it matters: every interpolation method downstream (Lagrange, linear,
    spline, PCHIP) assumes its input x-values (maturities) are sorted and
    distinct for a given curve. If this check ever fails, it means
    something upstream (Treasury's own data, or our tidying step) produced
    a curve out of order -- better to catch that here with a clear error
    than to have Lagrange silently misbehave on scrambled input.

    Hints:
    - Group `table` by "date".
    - Within each group, check `maturity_years` is strictly increasing
      (each value bigger than the last -- not just non-decreasing, since we
      already reject exact duplicates in validate_no_duplicate_observations).
    - pandas: a Series `s` is strictly increasing if `s.is_monotonic_increasing`
      is True AND `s` has no duplicates -- or check `(s.diff().dropna() > 0).all()`.
    - Raise ValueError naming the offending date if any group fails.
    """
    raise NotImplementedError("validate_maturity_ordering: implement me (see TODO above)")


def validate_curve_data(table: pd.DataFrame) -> None:
    """Run all validation checks. Raises on the first failure."""
    validate_required_columns(table)
    validate_no_duplicate_observations(table)
    validate_yield_range(table)
    validate_maturity_ordering(table)
