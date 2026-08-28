"""Feature spaces: the exact matrix a predictor consumes.

A model is only reproducible in production if the frame it is scored on is built
the same way as the frame it was trained on. Rather than leaving that to two
parallel code paths, the recipe is captured here as data, stored in the scoring
bundle, and replayed identically at both ends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .woe import WoeTransformer


@dataclass(frozen=True)
class FeatureSpace:
    """Which columns a model sees, and whether each is WOE-encoded or raw.

    - `woe_columns` are replaced by their weight of evidence, so they carry no
      missing values and are monotone in risk by construction.
    - `raw_columns` pass through untouched, missing values included, which lets a
      tree model find structure that binning would flatten.
    """

    name: str
    woe_columns: tuple[str, ...]
    raw_columns: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.raw_columns + self.woe_columns

    def build(self, df: pd.DataFrame, woe: WoeTransformer) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise KeyError(f"input frame is missing feature column(s): {missing}")
        data = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in self.raw_columns}
        for column in self.woe_columns:
            data[column] = woe.binnings[column].transform(df[column])
        return pd.DataFrame(data, index=df.index, columns=list(self.columns))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "woe_columns": list(self.woe_columns),
            "raw_columns": list(self.raw_columns),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FeatureSpace":
        return cls(
            name=data["name"],
            woe_columns=tuple(data["woe_columns"]),
            raw_columns=tuple(data["raw_columns"]),
        )


def woe_space(features: Sequence[str]) -> FeatureSpace:
    """Everything WOE-encoded — what a scorecard needs."""
    return FeatureSpace(name="woe", woe_columns=tuple(features), raw_columns=())


def mixed_space(features: Sequence[str], numeric: Sequence[str]) -> FeatureSpace:
    """Raw numerics plus WOE-encoded categoricals — what a tree model prefers."""
    numeric_set = set(numeric)
    raw = tuple(f for f in features if f in numeric_set)
    encoded = tuple(f for f in features if f not in numeric_set)
    return FeatureSpace(name="mixed", woe_columns=encoded, raw_columns=raw)
