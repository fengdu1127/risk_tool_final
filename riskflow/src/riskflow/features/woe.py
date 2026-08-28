"""WOE transformer: one fitted binning per feature, serialisable to JSON."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from ..data.schema import DatasetSchema
from ..logging_setup import get_logger
from ..settings import BinningSettings
from .binning import Binning, BinningError, bin_table, binning_from_dict, fit_binning

log = get_logger("woe")


@dataclass(frozen=True)
class WoeTransformer:
    binnings: dict[str, Binning]
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.binnings)

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        schema: DatasetSchema,
        settings: BinningSettings,
    ) -> "WoeTransformer":
        y = df[schema.label].astype(float)
        binnings: dict[str, Binning] = {}
        skipped: dict[str, str] = {}
        for feature in schema.features:
            try:
                binnings[feature] = fit_binning(
                    df[feature], y, schema.kind(feature), settings, feature
                )
            except BinningError as exc:
                skipped[feature] = str(exc)
            except Exception as exc:  # a broken column must not sink the run
                skipped[feature] = f"binning failed: {exc}"
        if skipped:
            log.warning("skipped %d feature(s) during binning: %s", len(skipped), sorted(skipped))
        if not binnings:
            raise ValueError("no feature could be binned; check the input data")
        log.info("fitted binnings for %d feature(s)", len(binnings))
        return cls(binnings=binnings, skipped=skipped)

    def transform(self, df: pd.DataFrame, features: tuple[str, ...] | None = None) -> pd.DataFrame:
        """WOE-encoded frame with one column per requested feature."""
        wanted = features or self.features
        missing = [f for f in wanted if f not in self.binnings]
        if missing:
            raise KeyError(f"no fitted binning for: {missing}")
        absent = [f for f in wanted if f not in df.columns]
        if absent:
            raise KeyError(f"input frame is missing feature column(s): {absent}")
        data = {f: self.binnings[f].transform(df[f]) for f in wanted}
        return pd.DataFrame(data, index=df.index, columns=list(wanted))

    def assign_bins(self, df: pd.DataFrame, features: tuple[str, ...] | None = None) -> pd.DataFrame:
        wanted = features or self.features
        data = {f: self.binnings[f].assign(df[f]) for f in wanted if f in df.columns}
        return pd.DataFrame(data, index=df.index)

    def iv_table(self) -> pd.DataFrame:
        rows = [
            {
                "feature": name,
                "kind": binning.kind,
                "iv": binning.iv,
                "n_bins": len(binning.stats),
                "strength": _iv_strength(binning.iv),
            }
            for name, binning in self.binnings.items()
        ]
        table = pd.DataFrame(rows)
        return table.sort_values("iv", ascending=False).reset_index(drop=True) if len(table) else table

    def bin_table(self) -> pd.DataFrame:
        return bin_table(self.binnings.values())

    def subset(self, features) -> "WoeTransformer":
        keep = [f for f in features if f in self.binnings]
        return WoeTransformer(binnings={f: self.binnings[f] for f in keep}, skipped=dict(self.skipped))

    def to_dict(self) -> dict:
        return {
            "binnings": {name: b.to_dict() for name, b in self.binnings.items()},
            "skipped": dict(self.skipped),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "WoeTransformer":
        return cls(
            binnings={name: binning_from_dict(spec) for name, spec in data["binnings"].items()},
            skipped=dict(data.get("skipped", {})),
        )


def _iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspicious"


def monotonic_correlation(binning: Binning) -> float:
    """Correlation between bin order and bad rate, ignoring the missing bin.

    Numeric bins are ordered by value, so this measures whether risk moves in one
    direction across the feature's range. Returns NaN for categorical features,
    where bin order carries no meaning.
    """
    if binning.kind != "numeric":
        return float("nan")
    stats = [s for s in binning.stats if s.index >= 0 and s.rows > 0]
    if len(stats) < 3:
        return float("nan")
    order = np.arange(len(stats), dtype=float)
    rates = np.array([s.bad_rate for s in stats], dtype=float)
    weights = np.array([s.rows for s in stats], dtype=float)
    return float(_weighted_corr(order, rates, weights))


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum()
    vy = (w * (y - my) ** 2).sum()
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / np.sqrt(vx * vy)
