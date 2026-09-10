"""Client for the U.S. Treasury's Daily Treasury Par Yield Curve Rates.

Data source: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/
Endpoint verified live on 2026-09-09 (see DECISIONS.md). One request per
calendar year -- that is how Treasury's endpoint is structured, not a
choice made here.

This module only knows how to talk to Treasury.gov. It raises on failure
rather than handling it; see loader.py for the graceful-fallback behavior
required by the project spec.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from utils.maturity import parse_maturity_label

_BASE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/{year}/all"
)

TIDY_COLUMNS = ["date", "maturity_years", "maturity_label", "yield_pct"]


def fetch_year(year: int, timeout: float = 30.0) -> pd.DataFrame:
    """Fetch one calendar year of daily par yield curve rates.

    Returns a tidy long-format table: one row per (date, maturity), columns
    ``date``, ``maturity_years``, ``maturity_label``, ``yield_pct``. A
    maturity with no quote on a given date (either it didn't exist yet, or
    simply wasn't published that day) is silently absent -- callers should
    not assume every date has every maturity.

    Raises ``requests.RequestException`` on any network/HTTP failure.
    """
    params = {
        "type": "daily_treasury_yield_curve",
        "field_tdr_date_value": str(year),
        "page": "",
        "_format": "csv",
    }
    response = requests.get(_BASE_URL.format(year=year), params=params, timeout=timeout)
    response.raise_for_status()
    wide = pd.read_csv(StringIO(response.text))
    return tidy_wide_table(wide)


def fetch_years(start_year: int, end_year: int, timeout: float = 30.0) -> pd.DataFrame:
    """Fetch and concatenate multiple calendar years (inclusive range)."""
    frames = [fetch_year(year, timeout=timeout) for year in range(start_year, end_year + 1)]
    return pd.concat(frames, ignore_index=True)


def tidy_wide_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert Treasury's wide CSV shape (one column per maturity) into the
    tidy long shape this project standardizes on everywhere else.

    Separated from ``fetch_year`` so it can be tested against a literal CSV
    snippet without any network access.
    """
    wide = wide.rename(columns={"Date": "date"})
    wide["date"] = pd.to_datetime(wide["date"], format="%m/%d/%Y")
    maturity_columns = [c for c in wide.columns if c != "date"]

    long = wide.melt(
        id_vars="date",
        value_vars=maturity_columns,
        var_name="maturity_label",
        value_name="yield_pct",
    )
    long = long.dropna(subset=["yield_pct"])
    long["maturity_years"] = long["maturity_label"].map(parse_maturity_label)
    return (
        long[TIDY_COLUMNS]
        .sort_values(["date", "maturity_years"])
        .reset_index(drop=True)
    )
