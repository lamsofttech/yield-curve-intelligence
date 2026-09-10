import numpy as np
import pandas as pd
import pytest

from models.pca_analysis import describe_loadings, fit_pca


def _synthetic_level_dominated_changes(n_days: int = 500, seed: int = 2026) -> pd.DataFrame:
    # Construct daily changes that are overwhelmingly a parallel shift
    # (same-sign, similar-magnitude across maturities) plus small noise --
    # PC1 on this data *should* come out same-sign across all columns.
    rng = np.random.default_rng(seed)
    maturities = [1 / 12, 1.0, 2.0, 5.0, 10.0, 30.0]
    level_shocks = rng.normal(0, 0.05, size=n_days)
    noise = rng.normal(0, 0.005, size=(n_days, len(maturities)))
    data = level_shocks[:, None] + noise
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(data, index=dates, columns=maturities)


def test_fit_pca_returns_requested_number_of_components():
    changes = _synthetic_level_dominated_changes()
    result = fit_pca(changes, n_components=3)
    assert result.loadings.shape == (6, 3)
    assert result.scores.shape == (len(changes), 3)
    assert len(result.explained_variance_ratio) == 3


def test_explained_variance_ratios_are_valid_proportions():
    result = fit_pca(_synthetic_level_dominated_changes())
    assert (result.explained_variance_ratio >= 0).all()
    assert (result.explained_variance_ratio <= 1).all()
    # Non-increasing by construction (PCA orders components by variance explained).
    ratios = result.explained_variance_ratio
    assert all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1))


def test_first_component_dominates_on_level_dominated_synthetic_data():
    # A sanity check on the synthetic construction itself: since the data
    # is deliberately built as ~95% common parallel-shift + small noise,
    # PC1 should explain the large majority of variance.
    result = fit_pca(_synthetic_level_dominated_changes())
    assert result.explained_variance_ratio[0] > 0.8


def test_describe_loadings_identifies_same_sign_pattern():
    result = fit_pca(_synthetic_level_dominated_changes())
    descriptions = describe_loadings(result.loadings)
    assert "same sign" in descriptions["PC1"]


def test_describe_loadings_identifies_a_slope_like_pattern():
    # Construct loadings by hand with a known monotonic sign change --
    # this tests describe_loadings' logic directly, independent of PCA.
    loadings = pd.DataFrame(
        {"PC1": [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]}, index=[1 / 12, 1.0, 2.0, 5.0, 10.0, 30.0]
    )
    descriptions = describe_loadings(loadings)
    assert "slope" in descriptions["PC1"] or "one sign change" in descriptions["PC1"]


def test_no_components_requested_beyond_available_maturities_raises():
    changes = _synthetic_level_dominated_changes()
    with pytest.raises(ValueError):
        fit_pca(changes, n_components=100)
