"""Points-based scorecard and the probability-to-score scale.

One scale serves both: `to_credit_score` and the scorecard's points sum to the
same number for the same applicant, so a reviewer can reconcile a decision line
by line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.woe import WoeTransformer
from ..settings import ScorecardSettings
from .predictors import LinearScorer


def scale_factors(settings: ScorecardSettings) -> tuple[float, float]:
    """(factor, offset) such that score = offset - factor * ln(odds_bad)."""
    factor = settings.pdo / np.log(2.0)
    offset = settings.base_score + factor * np.log(settings.base_odds)
    return float(factor), float(offset)


def credit_score_from_log_odds(log_odds, settings: ScorecardSettings) -> np.ndarray:
    """Map log-odds of going bad onto the credit-score scale — higher is safer.

    This is the definition. The score equals `base_score` exactly when the odds
    of going bad equal `base_odds`, and every `pdo` points doubles the
    good-to-bad odds.
    """
    factor, offset = scale_factors(settings)
    margin = np.asarray(pd.Series(log_odds).to_numpy(), dtype=float)
    return offset - factor * margin


def to_credit_score(probability, settings: ScorecardSettings) -> np.ndarray:
    """Credit score from a bad probability.

    Prefer `credit_score_from_log_odds` wherever the model's margin is available.
    A probability loses the information this scale needs at the extremes: past a
    log-odds of about 36 the sigmoid saturates to exactly 1.0 in float64, so
    every applicant beyond that point maps to the same score no matter how much
    riskier they get — collapsing the ordering precisely in the tail that
    matters most. The clip below keeps the arithmetic finite; it cannot recover
    the lost resolution.
    """
    p = np.clip(np.asarray(pd.Series(probability).to_numpy(), dtype=float), 1e-9, 1 - 1e-9)
    return credit_score_from_log_odds(np.log(p / (1.0 - p)), settings)


def build_scorecard(
    scorer: LinearScorer, woe: WoeTransformer, settings: ScorecardSettings
) -> pd.DataFrame:
    """Points per feature bin, plus a base row, summing to the credit score.

    Because WOE is ln(bad/good) here, a riskier bin has a higher WOE and so earns
    fewer points — the sign works out without a correction term.
    """
    factor, offset = scale_factors(settings)
    coefficients = dict(zip(scorer.features, scorer.coefficients))

    rows = [
        {
            "feature": "__base__",
            "bin": "",
            "label": "base points",
            "woe": 0.0,
            "coefficient": round(scorer.intercept, 6),
            "points": round(offset - factor * scorer.intercept, 2),
        }
    ]
    for feature, coefficient in coefficients.items():
        binning = woe.binnings.get(feature)
        if binning is None:
            continue
        for stat in binning.stats:
            rows.append(
                {
                    "feature": feature,
                    "bin": stat.index,
                    "label": stat.label,
                    "rows": stat.rows,
                    "bad_rate": stat.bad_rate,
                    "woe": stat.woe,
                    "coefficient": round(float(coefficient), 6),
                    "points": round(-factor * coefficient * stat.woe, 2),
                }
            )
    return pd.DataFrame(rows)


def verify_scorecard(
    scorecard: pd.DataFrame,
    scorer: LinearScorer,
    woe: WoeTransformer,
    df: pd.DataFrame,
    settings: ScorecardSettings,
) -> float:
    """Largest gap between summed scorecard points and the model's own score.

    A non-trivial gap means the card on the wall no longer matches the model in
    production, so this is asserted during training rather than trusted.
    """
    from ..features.space import woe_space

    encoded = woe_space(scorer.features).build(df, woe)
    # Compared against the margin, not the probability: the card sums log-odds,
    # so round-tripping through a saturating sigmoid would manufacture a
    # disagreement that does not exist.
    model_score = credit_score_from_log_odds(scorer.margin(encoded), settings)

    lookup = {
        (row["feature"], row["bin"]): row["points"]
        for _, row in scorecard.iterrows()
        if row["feature"] != "__base__"
    }
    base = float(scorecard.loc[scorecard["feature"] == "__base__", "points"].iloc[0])
    total = np.full(len(df), base, dtype=float)
    for feature in scorer.features:
        bins = woe.binnings[feature].assign(df[feature])
        total += np.array([lookup.get((feature, int(b)), 0.0) for b in bins], dtype=float)
    return float(np.max(np.abs(total - model_score)))
