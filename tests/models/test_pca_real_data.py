"""PCA sanity checks against the real bundled 2016-2026 historical data --
a "golden" regression test, distinct from the synthetic-data tests in
test_pca_analysis.py. Verified numerically before writing (see
DECISIONS.md / LEARNING_LOG.md, Phase 6): PC1=Level (72.6% var), PC2=Slope
(10.6%), PC3=Curvature (9.1%), reproducing the classic Litterman-Scheinkman
ordering from real Treasury data rather than assuming it.
"""

from data.loader import load_historical
from features.yield_changes import daily_changes, to_wide_panel
from models.pca_analysis import fit_pca, label_components


def _real_pca_result():
    tidy = load_historical()
    panel = to_wide_panel(tidy)
    changes = daily_changes(panel)
    return fit_pca(changes, n_components=3)


def test_three_components_explain_most_of_the_variance():
    result = _real_pca_result()
    assert sum(result.explained_variance_ratio) > 0.85


def test_first_component_is_dominant_and_labeled_level():
    result = _real_pca_result()
    assert result.explained_variance_ratio[0] > 0.5
    labels = label_components(result.loadings)
    assert labels["PC1"] == "Level"


def test_components_reproduce_the_classic_level_slope_curvature_ordering():
    result = _real_pca_result()
    labels = label_components(result.loadings)
    assert labels == {"PC1": "Level", "PC2": "Slope", "PC3": "Curvature"}


def test_variance_explained_is_strictly_decreasing():
    result = _real_pca_result()
    ratios = result.explained_variance_ratio
    assert ratios[0] > ratios[1] > ratios[2]
