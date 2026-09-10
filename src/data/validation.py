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

    Every interpolation method downstream (Lagrange, linear, spline, PCHIP)
    assumes its input x-values (maturities) are sorted and distinct for a
    given curve. If this ever fails, something upstream (Treasury's own
    data, or our tidying step) produced a curve out of order -- better to
    catch that here with a clear error than have Lagrange silently
    misbehave on scrambled input.
    """
    for observation_date, group in table.groupby("date"):
        maturities = group["maturity_years"]
        # Series.is_monotonic_increasing allows equal neighbors (it means
        # "non-decreasing"), so it would wrongly pass e.g. [1.0, 1.0, 2.0].
        # Checking every step is strictly positive enforces "strictly
        # increasing" without depending on validate_no_duplicate_observations
        # having already run first.
        if not (maturities.diff().dropna() > 0).all():
            raise ValueError(
                f"Maturities not strictly increasing for date {observation_date}: "
                f"{maturities.tolist()}"
            )


def validate_curve_data(table: pd.DataFrame) -> None:
    """Run all validation checks. Raises on the first failure."""
    validate_required_columns(table)
    validate_no_duplicate_observations(table)
    validate_yield_range(table)
    validate_maturity_ordering(table)
