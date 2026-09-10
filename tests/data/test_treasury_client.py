from io import StringIO

import pandas as pd
import pytest

from data.treasury_client import TIDY_COLUMNS, tidy_wide_table

# A literal snippet in Treasury's real wide shape, including a blank cell
# (the "4 Mo" column did not exist for all dates historically -- pandas
# reads an empty CSV field as NaN, which tidy_wide_table must drop).
_SAMPLE_CSV = """Date,"1 Mo","3 Mo","6 Mo","1 Yr","10 Yr"
09/08/2026,3.81,3.94,4.00,4.15,4.80
09/04/2026,3.79,3.91,3.98,4.13,4.78
"""

_SAMPLE_CSV_WITH_GAP = """Date,"1 Mo","3 Mo","4 Mo"
09/08/2026,3.81,3.94,
"""


def _read(csv_text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv_text))


def test_tidy_produces_expected_columns_and_row_count():
    long = tidy_wide_table(_read(_SAMPLE_CSV))
    assert list(long.columns) == TIDY_COLUMNS
    # 2 dates x 5 maturity columns = 10 rows
    assert len(long) == 10


def test_tidy_parses_dates_correctly():
    long = tidy_wide_table(_read(_SAMPLE_CSV))
    assert long["date"].max() == pd.Timestamp("2026-09-08")
    assert long["date"].min() == pd.Timestamp("2026-09-04")


def test_tidy_computes_maturity_years():
    long = tidy_wide_table(_read(_SAMPLE_CSV))
    row = long[(long["date"] == pd.Timestamp("2026-09-08")) & (long["maturity_label"] == "10 Yr")]
    assert row["maturity_years"].iloc[0] == pytest.approx(10.0)


def test_tidy_drops_missing_maturity_values():
    long = tidy_wide_table(_read(_SAMPLE_CSV_WITH_GAP))
    # "4 Mo" was blank for this date -- must not appear as a NaN yield row.
    assert not long["yield_pct"].isna().any()
    assert "4 Mo" not in long["maturity_label"].values


def test_tidy_sorts_by_date_then_maturity():
    long = tidy_wide_table(_read(_SAMPLE_CSV))
    first_date_rows = long[long["date"] == long["date"].min()]
    assert first_date_rows["maturity_years"].is_monotonic_increasing
