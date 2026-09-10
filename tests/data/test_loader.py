import pandas as pd
import requests

import data.loader as loader
from data.loader import CurveData, load_sample


def test_load_sample_reads_bundled_files():
    table = load_sample()
    assert len(table) > 0
    assert set(table.columns) == {"date", "maturity_years", "maturity_label", "yield_pct"}
    # Bundled sample data must be real, not fabricated -- at minimum it
    # should span more than a single day.
    assert table["date"].nunique() > 1


def test_load_curve_data_falls_back_to_sample_on_network_failure(monkeypatch):
    def broken_fetch_years(start_year, end_year, timeout=30.0):
        raise requests.exceptions.ConnectionError("simulated network failure")

    monkeypatch.setattr(loader, "fetch_years", broken_fetch_years)

    result = loader.load_curve_data(2025, 2026)

    assert isinstance(result, CurveData)
    assert result.source == "sample"
    assert result.error is not None
    assert "simulated network failure" in result.error
    assert result.as_of == result.table["date"].max()


def test_load_curve_data_uses_live_source_when_available(monkeypatch):
    fake_live_table = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02")],
            "maturity_years": [10.0],
            "maturity_label": ["10 Yr"],
            "yield_pct": [4.5],
        }
    )
    def fake_fetch_years(start_year, end_year, timeout=30.0):
        return fake_live_table

    monkeypatch.setattr(loader, "fetch_years", fake_fetch_years)

    result = loader.load_curve_data(2026, 2026)

    assert result.source == "live"
    assert result.error is None
    assert result.table.equals(fake_live_table)
