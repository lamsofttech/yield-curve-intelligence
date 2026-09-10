"""Parse Treasury maturity column labels (e.g. "1 Mo", "1.5 Month", "10 Yr")
into a maturity expressed in years as a float.

Deliberately a parser, not a fixed lookup table: Treasury has added new
short-tenor bill columns over time (2 Mo appeared ~2020, 4 Mo ~2022,
"1.5 Month" in 2026), and the unit spelling itself is inconsistent ("Mo"
for most columns, "Month" for the newest one). A lookup table would need
updating every time Treasury adds a tenor; this parser does not.
"""

from __future__ import annotations

import re

_LABEL_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(Mo|Month|Yr)\s*$", re.IGNORECASE)


def parse_maturity_label(label: str) -> float:
    """Convert one Treasury maturity label to maturity in years.

    Rounded to 10 decimal places -- far finer than needed to distinguish
    any two real maturities (the closest real pair, 1mo vs 1.5mo, differ by
    about 0.04), but enough to give a canonical value that survives a
    round-trip through CSV serialization unchanged. Without this, "1 Mo"
    parsed fresh (0.08333333333333333) does not exactly equal the same
    value read back from a CSV written by pandas.to_csv (which truncates
    to 15 significant digits, giving 0.0833333333333333) -- a real bug hit
    while building the Phase 6 historical panel; see LEARNING_LOG.md.

    >>> parse_maturity_label("1 Mo")
    0.0833333333
    >>> parse_maturity_label("1.5 Month")
    0.125
    >>> parse_maturity_label("10 Yr")
    10.0
    """
    match = _LABEL_PATTERN.match(label)
    if match is None:
        raise ValueError(f"Unrecognized maturity label: {label!r}")

    value = float(match.group(1))
    unit = match.group(2).lower()
    years = value / 12.0 if unit in ("mo", "month") else value
    return round(years, 10)
