"""Train / test / holdout construction.

The holdout is sealed: nothing in training, tuning, model selection or
calibration may touch it. It is opened once, at the end, to check that the
chosen model and policy still hold up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from ..settings import SplitSettings

log = get_logger("split")


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame
    strategy: str
    detail: dict

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [("train", self.train), ("test", self.test), ("holdout", self.holdout)]

    def profile(self, label: str, time_col: str | None = None) -> pd.DataFrame:
        rows = []
        for name, frame in self.items():
            row = {
                "dataset": name,
                "rows": len(frame),
                "bad_rate": round(float(frame[label].mean()), 6) if len(frame) else np.nan,
            }
            if time_col and time_col in frame.columns and len(frame):
                stamps = pd.to_datetime(frame[time_col], errors="coerce")
                row["time_min"] = stamps.min()
                row["time_max"] = stamps.max()
            rows.append(row)
        return pd.DataFrame(rows)


def split(df: pd.DataFrame, label: str, settings: SplitSettings) -> SplitResult:
    """Out-of-time holdout when a time column is configured, random otherwise."""
    if settings.time_col:
        return out_of_time_split(df, label, settings)
    return random_split(df, label, settings)


def random_split(df: pd.DataFrame, label: str, settings: SplitSettings) -> SplitResult:
    total = settings.train_ratio + settings.test_ratio + settings.holdout_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    rng = np.random.default_rng(settings.random_state)
    labels = df[label].to_numpy()
    positions = np.arange(len(df))

    # Stratify by shuffling positions within each class, so every split keeps the
    # base rate. Cut points are cumulative so rounding can never drop a row.
    parts: dict[str, list[np.ndarray]] = {"train": [], "test": [], "holdout": []}
    for value in pd.unique(pd.Series(labels).dropna()):
        idx = positions[labels == value]
        rng.shuffle(idx)
        n = len(idx)
        end_train = int(round(n * settings.train_ratio))
        end_test = int(round(n * (settings.train_ratio + settings.test_ratio)))
        end_test = max(end_test, end_train)
        parts["train"].append(idx[:end_train])
        parts["test"].append(idx[end_train:end_test])
        parts["holdout"].append(idx[end_test:])

    frames = {}
    for name, chunks in parts.items():
        taken = np.concatenate(chunks) if chunks else np.array([], dtype=int)
        rng.shuffle(taken)
        frames[name] = df.iloc[taken].reset_index(drop=True)

    if settings.holdout_ratio > 0:
        _guard_holdout(frames["holdout"], settings)
    log.info(
        "random stratified split | train=%d test=%d holdout=%d",
        len(frames["train"]), len(frames["test"]), len(frames["holdout"]),
    )
    return SplitResult(
        train=frames["train"],
        test=frames["test"],
        holdout=frames["holdout"],
        strategy="random_stratified",
        detail={"random_state": settings.random_state},
    )


def out_of_time_split(df: pd.DataFrame, label: str, settings: SplitSettings) -> SplitResult:
    """Newest `oot_months` become the holdout; older rows split randomly."""
    time_col = settings.time_col
    if not time_col:
        raise ValueError("out_of_time_split requires settings.time_col")
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in data")
    if settings.oot_months <= 0:
        raise ValueError("oot_months must be positive")

    frame = df.copy()
    stamps = pd.to_datetime(frame[time_col], errors="coerce")
    unparseable = int(stamps.isna().sum())
    if unparseable:
        log.warning("dropping %d rows with an unparseable %s", unparseable, time_col)
        frame = frame.loc[stamps.notna()]
        stamps = stamps.loc[stamps.notna()]
    if frame.empty:
        raise ValueError(f"no rows left after parsing '{time_col}'")

    cutoff = stamps.max() - pd.DateOffset(months=settings.oot_months)
    is_recent = stamps > cutoff
    holdout = frame.loc[is_recent].reset_index(drop=True)
    history = frame.loc[~is_recent]

    _guard_holdout(holdout, settings)
    if len(history) < 2 or history[label].nunique() < 2:
        raise ValueError(
            f"only {len(history)} usable historical rows before {cutoff.date()}; "
            "lower oot_months or widen the data window"
        )

    inner = SplitSettings(
        train_ratio=settings.train_ratio / (settings.train_ratio + settings.test_ratio),
        test_ratio=settings.test_ratio / (settings.train_ratio + settings.test_ratio),
        holdout_ratio=0.0,
        random_state=settings.random_state,
    )
    history_split = random_split(history.reset_index(drop=True), label, inner)

    log.info(
        "out-of-time split | train=%d test=%d holdout=%d (holdout starts after %s)",
        len(history_split.train), len(history_split.test), len(holdout), cutoff.date(),
    )
    return SplitResult(
        train=history_split.train,
        test=history_split.test,
        holdout=holdout,
        strategy="out_of_time",
        detail={
            "time_col": time_col,
            "oot_months": settings.oot_months,
            "holdout_starts_after": str(cutoff),
            "dropped_unparseable_rows": unparseable,
        },
    )


def _guard_holdout(holdout: pd.DataFrame, settings: SplitSettings) -> None:
    if len(holdout) < settings.min_holdout_rows:
        raise ValueError(
            f"holdout has {len(holdout)} rows, below min_holdout_rows="
            f"{settings.min_holdout_rows}; it is too small to validate against"
        )
