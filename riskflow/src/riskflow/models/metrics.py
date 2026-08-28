"""Discrimination metrics, implemented in numpy.

These run on both sides of the fence — training reports and the scoring service's
self-checks — so they deliberately avoid scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y_true).to_numpy(), dtype=float)
    s = np.asarray(pd.Series(y_score).to_numpy(), dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"label and score lengths differ: {y.shape} vs {s.shape}")
    keep = ~(np.isnan(y) | np.isnan(s))
    return y[keep], s[keep]


def auc(y_true, y_score) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic (ties averaged)."""
    y, s = _clean(y_true, y_score)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Average the ranks inside each group of tied scores.
    sorted_scores = s[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks(y_true, y_score) -> float:
    """Kolmogorov-Smirnov separation between the bad and good score distributions."""
    y, s = _clean(y_true, y_score)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    # Only compare at the last row of each tied block; a threshold cannot split ties.
    boundary = np.append(s_sorted[1:] != s_sorted[:-1], True)
    return float(np.max(np.abs(tpr[boundary] - fpr[boundary])))


def gini(y_true, y_score) -> float:
    return 2.0 * auc(y_true, y_score) - 1.0


def lift(y_true, flagged) -> float:
    """Bad rate among flagged rows, relative to the overall bad rate."""
    y = np.asarray(pd.Series(y_true).to_numpy(), dtype=float)
    mask = np.asarray(pd.Series(flagged).to_numpy(), dtype=bool)
    base = y.mean() if len(y) else 0.0
    if mask.sum() == 0 or base == 0:
        return 0.0
    return float(y[mask].mean() / base)


def summary(y_true, y_score, dataset: str = "") -> dict:
    return {
        "dataset": dataset,
        "rows": int(len(pd.Series(y_true))),
        "bad_rate": round(float(pd.Series(y_true).mean()), 6),
        "auc": round(auc(y_true, y_score), 6),
        "ks": round(ks(y_true, y_score), 6),
        "gini": round(gini(y_true, y_score), 6),
    }


def gains_table(
    y_true,
    y_score,
    dataset: str = "",
    n_bands: int = 10,
    edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bad rate, lift and cumulative bad capture per score band, riskiest first.

    Pass `edges` from the training sample to hold the band boundaries fixed, so
    bands mean the same thing across train, test and holdout.
    """
    y, s = _clean(y_true, y_score)
    if len(y) == 0:
        return pd.DataFrame()
    if edges is None:
        edges = band_edges(s, n_bands)
    band = np.searchsorted(edges, s, side="left")
    band = np.clip(band, 0, len(edges))

    overall = y.mean()
    total_bad = y.sum()
    rows = []
    for b in sorted(set(band.tolist()), reverse=True):
        mask = band == b
        count = int(mask.sum())
        bads = float(y[mask].sum())
        rows.append(
            {
                "dataset": dataset,
                "band": int(b),
                "rows": count,
                "bads": int(bads),
                "bad_rate": round(bads / count, 6) if count else 0.0,
                "lift": round((bads / count) / overall, 6) if count and overall else 0.0,
                "score_min": round(float(s[mask].min()), 6),
                "score_max": round(float(s[mask].max()), 6),
            }
        )
    table = pd.DataFrame(rows)
    table["share"] = (table["rows"] / len(y)).round(6)
    table["cum_share"] = table["share"].cumsum().round(6)
    table["cum_bad_capture"] = (table["bads"].cumsum() / total_bad).round(6) if total_bad else 0.0
    table["bad_rate_monotone"] = bool(table["bad_rate"].is_monotonic_decreasing)
    return table.reset_index(drop=True)


def band_edges(scores, n_bands: int = 10) -> np.ndarray:
    """Interior quantile boundaries of a score distribution."""
    s = np.asarray(pd.Series(scores).dropna().to_numpy(), dtype=float)
    if s.size == 0:
        return np.array([])
    quantiles = np.linspace(0, 1, n_bands + 1)[1:-1]
    return np.unique(np.quantile(s, quantiles))
