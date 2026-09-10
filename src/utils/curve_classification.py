"""Yield curve shape classification: normal, inverted, flat, or mixed.

Rule, stated explicitly (surfaced in the app so the user knows exactly
what it means, not left as a black-box label):

- "Inverted"  if both 10Y-2Y and 10Y-3M spreads are below -flat_threshold.
- "Normal"    if both spreads are above +flat_threshold.
- "Flat"      if both spreads are within +/-flat_threshold of zero.
- "Mixed"     otherwise -- the two spreads disagree (one says inverted-ish,
              the other doesn't), which is itself informative: it means
              the short end and long end of the curve are not moving
              together.
"""

from __future__ import annotations

flat_threshold_default = 0.10  # percentage points


def classify_curve(spread_10y_2y: float, spread_10y_3m: float, flat_threshold: float = 0.10) -> str:
    if spread_10y_2y < -flat_threshold and spread_10y_3m < -flat_threshold:
        return "Inverted"
    if spread_10y_2y > flat_threshold and spread_10y_3m > flat_threshold:
        return "Normal"
    if abs(spread_10y_2y) <= flat_threshold and abs(spread_10y_3m) <= flat_threshold:
        return "Flat"
    return "Mixed"


def compute_spreads(curve: dict[float, float]) -> tuple[float, float]:
    """Compute (10Y-2Y, 10Y-3M) spreads from a {maturity_years: yield_pct} curve.

    Raises KeyError with a clear message if a required maturity is absent
    (e.g. an offline sample date that happens not to include one of these).
    """
    required = {10.0: "10Yr", 2.0: "2Yr", 0.25: "3Mo"}
    missing = [label for maturity, label in required.items() if maturity not in curve]
    if missing:
        raise KeyError(f"Cannot compute spreads: missing maturities {missing}")
    return curve[10.0] - curve[2.0], curve[10.0] - curve[0.25]
