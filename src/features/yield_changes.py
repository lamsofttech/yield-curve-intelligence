"""Feature engineering: pivot tidy yield data into a wide daily panel and
compute day-over-day yield changes.

Restricted to a fixed set of maturities present throughout the entire
historical window, not "whatever columns exist that year" -- the Treasury
maturity grid itself has grown over time (see DECISIONS.md), so a
consistent multi-year panel needs a common subset, or PCA would be
comparing curves with different numbers of dimensions across years.
"""

from __future__ import annotations

import pandas as pd

from utils.maturity import parse_maturity_label

# Present in Treasury's published grid continuously since at least 2016.
# 2 Mo (~2020), 4 Mo (~2022), and 1.5 Month (2026) were all added later and
# are excluded here so every row of the panel has the same columns
# regardless of which year it came from.
#
# Derived from the labels via parse_maturity_label rather than written as
# separate float literals (e.g. 1 / 12) -- a second hand-written source of
# "the same" value is exactly how the CSV round-trip precision bug in
# LEARNING_LOG.md happened: two independently-computed values that *should*
# be identical but weren't, bit-for-bit. One source of truth avoids that
# class of bug entirely rather than just papering over this one instance.
COMMON_MATURITY_LABELS = [
    "1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]
COMMON_MATURITIES = [parse_maturity_label(label) for label in COMMON_MATURITY_LABELS]


def to_wide_panel(tidy: pd.DataFrame, maturities: list[float] | None = None) -> pd.DataFrame:
    """Pivot the tidy long-format table into a wide date x maturity panel.

    Restricted to `maturities` (default COMMON_MATURITIES) so the result
    has identical columns for every row. A date missing any of the
    requested maturities is dropped entirely (not imputed) -- an
    interpolated *feature* silently standing in for a missing observation
    would be a form of look-ahead/fabrication in exactly the kind of model
    this project is supposed to be honest about.
    """
    maturities = COMMON_MATURITIES if maturities is None else maturities
    wide = tidy.pivot(index="date", columns="maturity_years", values="yield_pct")
    missing = [m for m in maturities if m not in wide.columns]
    if missing:
        raise ValueError(f"Requested maturities not present in data: {missing}")
    return wide[maturities].sort_index().dropna()


def daily_changes(wide_panel: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day yield changes (today's yield minus the prior day's).

    Same columns as the input panel; the first row is dropped (no prior
    day exists to difference against).
    """
    return wide_panel.diff().dropna()
