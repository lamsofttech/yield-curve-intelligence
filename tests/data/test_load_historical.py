import pandas as pd

from data.loader import load_historical


def test_load_historical_returns_multiyear_real_data():
    table = load_historical()
    assert len(table) > 10_000  # 2016-2026 daily data across 11+ maturities
    assert set(table.columns) == {"date", "maturity_years", "maturity_label", "yield_pct"}
    assert table["date"].min() < pd.Timestamp("2017-01-01")
    assert table["date"].max() > pd.Timestamp("2025-01-01")
