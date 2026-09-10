"""PCA on daily Treasury yield changes.

Deliberately NOT standardized (no unit-variance scaling) before PCA. This
is a specific modeling choice, not an oversight: standard practice in
fixed-income term-structure analysis (Litterman & Scheinkman, 1991) runs
PCA on the raw covariance of yield changes, in the same units (percentage
points) across all maturities. The whole point of the first component
("level") is that it represents a genuinely comparable-sized move across
the curve; standardizing to unit variance would artificially equalize
short-end volatility (which is naturally much higher day-to-day) against
long-end volatility, and distort that interpretation.

Component labels (level/slope/curvature) are assigned only after
inspecting the actual fitted loadings -- never assumed from component
order alone, since sign is arbitrary (a component and its negation
represent the same underlying factor) and shape can vary. See
inspect_loadings() and DECISIONS.md for what was actually observed on
real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAResult:
    model: PCA
    loadings: pd.DataFrame  # index: maturity, columns: PC1, PC2, PC3, ...
    scores: pd.DataFrame  # index: date, columns: PC1, PC2, PC3, ...
    explained_variance_ratio: np.ndarray


def fit_pca(daily_change_panel: pd.DataFrame, n_components: int = 3) -> PCAResult:
    """Fit PCA on a wide date x maturity panel of daily yield changes.

    n_components=3 by default: this project's stated scope is level/slope/
    curvature (the first three factors), not a full spectral decomposition.
    """
    model = PCA(n_components=n_components)
    scores = model.fit_transform(daily_change_panel.to_numpy())

    component_names = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(
        model.components_.T, index=daily_change_panel.columns, columns=component_names
    )
    scores_df = pd.DataFrame(scores, index=daily_change_panel.index, columns=component_names)

    return PCAResult(
        model=model,
        loadings=loadings,
        scores=scores_df,
        explained_variance_ratio=model.explained_variance_ratio_,
    )


def label_components(loadings: pd.DataFrame) -> dict[str, str]:
    """Assign level/slope/curvature labels based on the actual observed
    loading shape (via describe_loadings), not component order or any
    assumption about which PC "should" be which.

    Verified on the real 2016-2026 history (see DECISIONS.md): PC1 same-
    sign (level), PC2 one sign change (slope), PC3 two sign changes
    (curvature) -- this is the classic ordering (Litterman & Scheinkman,
    1991), reproduced here from real data rather than assumed. A component
    with a shape not matching any of the three patterns is labeled
    "unclassified" rather than forced into one of the three -- and note
    that a component's *sign* is arbitrary (PCA could equally have
    returned every loading in a column negated, representing the exact
    same factor) - only the *shape* of the loadings is used here.
    """
    descriptions = describe_loadings(loadings)
    labels = {}
    for component, description in descriptions.items():
        if "parallel-shift-like" in description:
            labels[component] = "Level"
        elif "slope-like" in description:
            labels[component] = "Slope"
        elif "curvature-like" in description:
            labels[component] = "Curvature"
        else:
            labels[component] = "Unclassified"
    return labels


def describe_loadings(loadings: pd.DataFrame) -> dict[str, str]:
    """Produce a human-readable shape description for each component's
    loadings, WITHOUT assuming which component is "level"/"slope"/
    "curvature" -- that labeling is a judgment call made by inspecting
    this output, not something inferred automatically here.
    """
    descriptions = {}
    for column in loadings.columns:
        values = loadings[column].to_numpy()
        signs = np.sign(values)
        if np.all(signs == signs[0]):
            shape = "same sign across all maturities (parallel-shift-like)"
        elif np.sum(np.diff(signs) != 0) == 1:
            shape = "one sign change (monotonic short-vs-long pattern, slope-like)"
        else:
            shape = f"{np.sum(np.diff(signs) != 0)} sign changes (non-monotonic, curvature-like)"
        descriptions[column] = shape
    return descriptions
