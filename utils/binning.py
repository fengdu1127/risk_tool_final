"""Shared binning and WOE computation, used by EDA, model, and strategy modules.

All modules must produce identical cut points for the same feature so that
EDA reports, the WOE encoder, and strategy monotonicity checks stay consistent.
"""
import numpy as np
import pandas as pd

WOE_FLOOR = 1e-6


def tree_bin_thresholds(
    col: pd.Series,
    y: pd.Series,
    bins: int = 10,
    min_samples_leaf: int = 50,
    random_state: int = 42,
) -> list:
    """Supervised binning: thresholds from a depth-limited decision tree.

    `col` must be numeric with missing values already filled by the caller.
    """
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier(
        max_leaf_nodes=bins,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    dt.fit(np.asarray(col, dtype=float).reshape(-1, 1), y)
    return sorted(set(dt.tree_.threshold[dt.tree_.threshold != -2]))


def woe_stats(bin_series: pd.Series, y: pd.Series, floor: float = WOE_FLOOR) -> pd.DataFrame:
    """Per-bin bad/good counts, distributions, and WOE = ln(good_pct / bad_pct).

    Index is the bin id/category; rows with a null bin are dropped.
    """
    tmp = pd.DataFrame({"bin": bin_series, "label": y}).dropna(subset=["bin"])
    grouped = tmp.groupby("bin")["label"].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = max(grouped["bad"].sum(), floor)
    total_good = max(grouped["good"].sum(), floor)
    grouped["bad_pct"] = (grouped["bad"] / total_bad).replace(0, floor)
    grouped["good_pct"] = (grouped["good"] / total_good).replace(0, floor)
    grouped["woe"] = np.log(grouped["good_pct"] / grouped["bad_pct"])
    grouped["bad_rate"] = grouped["bad"] / grouped["total"]
    return grouped
