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
