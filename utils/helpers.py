"""Shared helpers for data splitting, metrics, reports, and persistence."""
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def get_logger(name: str = "risk_tool") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def dataset_profile(df: pd.DataFrame, label_col: str, time_col: str | None = None) -> dict:
    profile = {
        "n": int(len(df)),
        "bad_rate": round(float(df[label_col].mean()), 6) if len(df) else np.nan,
    }
    if time_col and time_col in df.columns and len(df):
        times = pd.to_datetime(df[time_col], errors="coerce")
        profile["time_min"] = times.min()
        profile["time_max"] = times.max()
    return profile


def split_dataset(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float = 0.70,
    test_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    time_col: str = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy split: train/test/holdout by time order or random stratification."""
    assert abs(train_ratio + test_ratio + holdout_ratio - 1.0) < 1e-6, (
        "train/test/holdout ratios must sum to 1"
    )

    if time_col and time_col in df.columns:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        n = len(df_sorted)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)
        train_df = df_sorted.iloc[:n_train]
        test_df = df_sorted.iloc[n_train: n_train + n_test]
        holdout_df = df_sorted.iloc[n_train + n_test:]
        logger.info(
            f"time split | train={len(train_df)}, test={len(test_df)}, holdout={len(holdout_df)}"
        )
    else:
        from sklearn.model_selection import train_test_split

        holdout_size = holdout_ratio
        train_test_size = 1 - holdout_size
        test_size_in_train = test_ratio / train_test_size

        train_test_df, holdout_df = train_test_split(
            df,
            test_size=holdout_size,
            stratify=df[label_col],
            random_state=random_state,
        )
        train_df, test_df = train_test_split(
            train_test_df,
            test_size=test_size_in_train,
            stratify=train_test_df[label_col],
            random_state=random_state,
        )
        logger.info(
            f"random stratified split | train={len(train_df)}, test={len(test_df)}, holdout={len(holdout_df)}"
        )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def split_dataset_with_recent_valid(
    df: pd.DataFrame,
    label_col: str,
    time_col: str,
    valid_months: int = 3,
    train_ratio: float = 0.70,
    test_ratio: float = 0.15,
    random_state: int = 42,
    min_valid_samples: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Use latest N months as valid, then random-stratify earlier rows into train/test."""
    if not time_col:
        raise ValueError("time_col is required for recent-valid split")
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' does not exist in data")
    if valid_months <= 0:
        raise ValueError("valid_months must be positive")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    missing_time = int(out[time_col].isna().sum())
    if missing_time:
        raise ValueError(f"time_col '{time_col}' has {missing_time} unparseable or missing values")

    max_time = out[time_col].max()
    valid_start = max_time - pd.DateOffset(months=valid_months)
    valid_df = out[out[time_col] > valid_start].copy()
    history_df = out[out[time_col] <= valid_start].copy()

    if len(valid_df) < min_valid_samples:
        raise ValueError(
            f"valid set has only {len(valid_df)} rows, below min_valid_samples={min_valid_samples}"
        )
    if len(history_df) < 2:
        raise ValueError("not enough historical rows left for train/test split")

    from sklearn.model_selection import train_test_split

    history_total = train_ratio + test_ratio
    test_size = test_ratio / history_total if history_total > 0 else 0.2
    stratify = history_df[label_col] if history_df[label_col].nunique() == 2 else None
    train_df, test_df = train_test_split(
        history_df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    split_info = {
        "time_col": time_col,
        "valid_months": valid_months,
        "valid_start": valid_start,
        "max_time": max_time,
        "missing_time_count": missing_time,
        "train": dataset_profile(train_df, label_col, time_col),
        "test": dataset_profile(test_df, label_col, time_col),
        "valid": dataset_profile(valid_df, label_col, time_col),
    }
    logger.info(
        f"recent-valid split | train={len(train_df)}, test={len(test_df)}, valid={len(valid_df)}, "
        f"valid_start>{valid_start.date()}"
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        split_info,
    )


def calc_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def calc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_prob))


def calc_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate population stability index."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    expected = expected[~pd.isna(expected)]
    actual = actual[~pd.isna(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    def _pct(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        pct = counts / len(arr)
        pct = np.where(pct == 0, 1e-6, pct)
        return pct

    exp_pct = _pct(expected)
    act_pct = _pct(actual)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def calc_lift(y_true: np.ndarray, y_pred_flag: np.ndarray) -> float:
    overall_bad_rate = y_true.mean()
    mask = y_pred_flag.astype(bool)
    if mask.sum() == 0 or overall_bad_rate == 0:
        return 0.0
    return float(y_true[mask].mean() / overall_bad_rate)


def model_report(y_true, y_prob, dataset_name: str = "") -> dict:
    auc = calc_auc(y_true, y_prob)
    ks = calc_ks(y_true, y_prob)
    gini = 2 * auc - 1
    return {"dataset": dataset_name, "AUC": round(auc, 4), "KS": round(ks, 4), "Gini": round(gini, 4)}


def _monotone_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted monotone smoothing for high-score-to-low-score bin rates."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        return values
    try:
        from sklearn.isotonic import IsotonicRegression

        x = np.arange(len(values))
        fitted = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(x, values, sample_weight=weights)
        return np.asarray(fitted, dtype=float)
    except Exception:
        return np.minimum.accumulate(values)


def score_bin_report(y_true, y_prob, dataset_name: str, n_bins: int = 10, breakpoints=None) -> pd.DataFrame:
    tmp = pd.DataFrame({"label": y_true, "score": y_prob}).dropna()
    if tmp.empty:
        return pd.DataFrame()
    if breakpoints is None:
        breakpoints = np.unique(np.quantile(tmp["score"], np.linspace(0, 1, n_bins + 1)))
        if len(breakpoints) >= 2:
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf
    if len(breakpoints) < 2:
        return pd.DataFrame()
    tmp["score_bin"] = pd.cut(tmp["score"], bins=breakpoints, labels=False, include_lowest=True, duplicates="drop")
    tmp = tmp.dropna(subset=["score_bin"])
    tmp["score_bin"] = tmp["score_bin"].astype(int)
    overall_bad = tmp["label"].mean()
    grouped = tmp.groupby("score_bin", dropna=False).agg(
        count=("label", "size"),
        bad_count=("label", "sum"),
        score_min=("score", "min"),
        score_max=("score", "max"),
    ).reset_index()
    intervals = []
    for bin_id in grouped["score_bin"]:
        left = breakpoints[int(bin_id)]
        right = breakpoints[int(bin_id) + 1]
        left_label = "-inf" if np.isneginf(left) else f"{left:.6f}"
        right_label = "inf" if np.isposinf(right) else f"{right:.6f}"
        intervals.append(f"({left_label}, {right_label}]")
    grouped["dataset"] = dataset_name
    grouped["score_bin_interval"] = intervals
    grouped["raw_bad_rate"] = grouped["bad_count"] / grouped["count"]
    grouped["bad_rate"] = grouped["raw_bad_rate"]
    grouped["lift"] = grouped["bad_rate"] / overall_bad if overall_bad else 0.0
    grouped = grouped.sort_values("score_bin", ascending=False).reset_index(drop=True)
    grouped["bin_order"] = range(1, len(grouped) + 1)
    grouped["raw_bad_rate_monotone"] = bool(grouped["raw_bad_rate"].is_monotonic_decreasing)
    grouped["monotone_bad_rate"] = _monotone_decreasing(grouped["raw_bad_rate"].values, grouped["count"].values)
    grouped["monotone_lift"] = grouped["monotone_bad_rate"] / overall_bad if overall_bad else 0.0
    grouped["bad_rate_monotone"] = bool(pd.Series(grouped["monotone_bad_rate"]).is_monotonic_decreasing)
    total_bad = grouped["bad_count"].sum()
    grouped["cum_bad_capture"] = grouped["bad_count"].cumsum() / total_bad if total_bad else 0.0
    cols = [
            "dataset",
            "score_bin",
            "bin_order",
            "score_bin_interval",
            "count",
            "bad_count",
            "raw_bad_rate",
            "bad_rate",
            "monotone_bad_rate",
            "bad_rate_monotone",
            "raw_bad_rate_monotone",
            "lift",
            "monotone_lift",
            "cum_bad_capture",
        "score_min",
        "score_max",
    ]
    return grouped[cols]


def feature_psi_report(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    features: list[str],
    dataset_name: str,
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in expected_df.columns or feature not in actual_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(expected_df[feature]):
            continue
        try:
            psi = calc_psi(expected_df[feature].values, actual_df[feature].values, bins=bins)
            rows.append({"dataset": dataset_name, "feature": feature, "PSI": round(psi, 6)})
        except Exception as exc:
            rows.append({"dataset": dataset_name, "feature": feature, "PSI": np.nan, "error": str(exc)})
    return pd.DataFrame(rows)


def save_pickle(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"saved {path}")


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"saved {path}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_inputs(df: pd.DataFrame, label_col: str, feature_cols: list = None):
    issues = []
    n = len(df)
    if n < 100:
        issues.append(f"sample size is small: {n}")

    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' does not exist. Available columns: {list(df.columns)}")

    label_vals = df[label_col].dropna().unique()
    if len(label_vals) < 2:
        raise ValueError(f"label_col '{label_col}' has one class only: {label_vals}")
    if len(label_vals) > 10:
        logger.warning(f"label_col '{label_col}' has {len(label_vals)} values; binary 0/1 is expected")

    label_counts = df[label_col].value_counts()
    minority_pct = label_counts.min() / label_counts.sum()
    if minority_pct < 0.01:
        logger.warning(f"minority class rate is only {minority_pct:.2%}; modeling may be unstable")
    if minority_pct > 0.45:
        logger.info(f"label distribution is balanced; minority class={minority_pct:.1%}")

    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns do not exist: {missing}")
        for c in feature_cols:
            if c in df.columns and df[c].nunique() <= 1:
                issues.append(f"feature '{c}' has zero variance")
    else:
        num_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 0:
            raise ValueError("data has no numeric feature columns")

    if issues:
        logger.warning("input data quality issues:\n  " + "\n  ".join(issues))
    return True
