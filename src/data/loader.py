"""Loads Treasury par yield curve data with graceful fallback to bundled
sample data if the live Treasury.gov source is unavailable.

The app must never crash outright just because an external HTTP call
failed -- it should show the user real (if older) data and say so clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from data.treasury_client import fetch_years, tidy_wide_table

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"
HISTORICAL_DIR = Path(__file__).resolve().parents[2] / "data" / "historical"


@dataclass(frozen=True)
class CurveData:
    """Yield curve data plus provenance.

    The app must always be able to tell the user whether they are looking
    at live data or the bundled fallback, and as of what date -- never
    presenting either silently as if it were the other.
    """

    table: pd.DataFrame  # tidy long format: date, maturity_years, maturity_label, yield_pct
    source: Literal["live", "sample"]
    as_of: pd.Timestamp
    error: str | None = None  # populated when source == "sample" because live fetch failed


def load_curve_data(start_year: int, end_year: int) -> CurveData:
    """Try the live Treasury.gov source first; fall back to the bundled
    sample data on any network/HTTP failure. Never raises.
    """
    try:
        table = fetch_years(start_year, end_year)
        return CurveData(table=table, source="live", as_of=table["date"].max())
    except requests.RequestException as exc:
        table = load_sample()
        return CurveData(table=table, source="sample", as_of=table["date"].max(), error=str(exc))


def load_sample() -> pd.DataFrame:
    """Load and tidy every bundled sample CSV in data/sample/.

    Sample files are stored in Treasury's original wide format (one column
    per maturity), the same shape the live endpoint returns, so they go
    through the identical ``tidy_wide_table`` parsing path as live data --
    no separate, untested code path just for the offline case.
    """
    csv_paths = sorted(SAMPLE_DIR.glob("*.csv"))
    frames = [tidy_wide_table(pd.read_csv(csv_path)) for csv_path in csv_paths]
    if not frames:
        raise FileNotFoundError(f"No bundled sample data found in {SAMPLE_DIR}")
    return pd.concat(frames, ignore_index=True)


def load_historical() -> pd.DataFrame:
    """Load the bundled multi-year historical dataset (data/historical/)
    used by the PCA and anomaly-detection features.

    Deliberately bundled/committed rather than fetched live at app
    startup: a live 2016-2026 fetch takes several minutes (11 sequential
    year-requests to Treasury.gov -- measured, not estimated), which is a
    bad cold-start experience for a Space, and PCA/anomaly detection are
    inherently backward-looking analyses that do not need to reflect data
    from the last few minutes to be meaningful. The Market Overview and
    Interpolation Lab sections still use load_curve_data() for the latest
    single-day curve, live with graceful fallback.

    Unlike data/sample/ (Treasury's original wide CSV shape), this file is
    stored already tidy (long format, with maturity_years pre-computed) --
    see scripts or DECISIONS.md for how it was built.
    """
    csv_paths = sorted(HISTORICAL_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No bundled historical data found in {HISTORICAL_DIR}")
    frames = [pd.read_csv(csv_path, parse_dates=["date"]) for csv_path in csv_paths]
    return pd.concat(frames, ignore_index=True)
