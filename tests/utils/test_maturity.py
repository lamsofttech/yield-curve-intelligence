import pytest

from utils.maturity import parse_maturity_label


@pytest.mark.parametrize(
    ("label", "expected_years"),
    [
        ("1 Mo", 1 / 12),
        ("1.5 Month", 1.5 / 12),
        ("2 Mo", 2 / 12),
        ("3 Mo", 3 / 12),
        ("4 Mo", 4 / 12),
        ("6 Mo", 6 / 12),
        ("1 Yr", 1.0),
        ("2 Yr", 2.0),
        ("10 Yr", 10.0),
        ("30 Yr", 30.0),
    ],
)
def test_parses_every_real_treasury_label(label, expected_years):
    assert parse_maturity_label(label) == pytest.approx(expected_years)


def test_is_case_insensitive_on_unit():
    assert parse_maturity_label("3 mo") == pytest.approx(0.25)
    assert parse_maturity_label("3 MO") == pytest.approx(0.25)
    assert parse_maturity_label("3 yr") == pytest.approx(3.0)


def test_rejects_unrecognized_label():
    with pytest.raises(ValueError, match="Unrecognized maturity label"):
        parse_maturity_label("not a maturity")


def test_rejects_missing_unit():
    with pytest.raises(ValueError):
        parse_maturity_label("10")
