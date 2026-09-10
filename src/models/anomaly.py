"""Anomaly detection over historical daily yield curves.

Uses Isolation Forest on the standardized daily yield-LEVEL panel (each
row = one day's full curve shape, one column per maturity) -- distinct
from the PCA module, which analyzes daily *changes*. This module answers
"how unusual is this day's curve *shape/level* relative to history?", not
"how unusual was today's *move*?".

Standardization here (unlike pca_analysis.py) is reasonable: Isolation
Forest splits are chosen per-feature independently, so a maturity with a
naturally wider historical range does not need to dominate the isolation
paths -- but this is a much weaker requirement than for distance-based
methods, since IsolationForest's splits are threshold-based, not
distance-based.

An anomaly score is a statistical statement about historical rarity, not
a judgment that anything is wrong, risky, or a trading opportunity -- see
the responsible-use statement surfaced with these results in the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AnomalyResult:
    scores: pd.Series  # index: date, values: anomaly score (lower = more unusual)
    is_anomaly: pd.Series  # index: date, values: bool (sklearn's -1/1 prediction, as bool)


def fit_anomaly_detector(
    level_panel: pd.DataFrame, contamination: float = 0.05, random_state: int = 2026
) -> AnomalyResult:
    """Fit Isolation Forest on a wide date x maturity panel of yield LEVELS.

    contamination=0.05 is sklearn's mechanism for "what fraction of days do
    we expect to be flagged" -- it does not mean 5% of days are provably
    anomalous in any absolute sense, only that the threshold is calibrated
    to flag roughly that fraction of the *given* historical sample.
    random_state is fixed for reproducibility, per this project's standard.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(level_panel.to_numpy())

    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(scaled)  # 1 = normal, -1 = anomaly
    raw_scores = model.score_samples(scaled)  # higher = more normal

    return AnomalyResult(
        scores=pd.Series(raw_scores, index=level_panel.index, name="anomaly_score"),
        is_anomaly=pd.Series(predictions == -1, index=level_panel.index, name="is_anomaly"),
    )


def most_unusual_dates(result: AnomalyResult, n: int = 10) -> pd.DataFrame:
    """The n most unusual dates by anomaly score, most unusual first."""
    combined = pd.DataFrame({"anomaly_score": result.scores, "is_anomaly": result.is_anomaly})
    return combined.sort_values("anomaly_score").head(n)


def current_percentile(result: AnomalyResult, as_of: pd.Timestamp) -> float:
    """Where the most recent date's anomaly score ranks against history, as
    a percentile (0 = most unusual ever seen, 100 = most typical).
    """
    if as_of not in result.scores.index:
        raise ValueError(f"{as_of} not found in anomaly scores")
    return float((result.scores <= result.scores.loc[as_of]).mean() * 100)
