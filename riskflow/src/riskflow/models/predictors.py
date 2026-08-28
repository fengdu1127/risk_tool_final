"""Serialisable predictors that evaluate with numpy alone.

Training may use scikit-learn or XGBoost, but whatever it produces is exported
into one of the structures below and stored as plain JSON. The scoring service
then needs neither library, which removes the usual "the pickle was written by a
different sklearn" failure mode entirely: a bundle stays readable as long as
numpy can multiply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Predictor(Protocol):
    kind: str
    features: tuple[str, ...]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def to_dict(self) -> dict: ...


def _matrix(X: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise KeyError(f"scoring frame is missing model input(s): {missing}")
    return X.loc[:, list(features)].to_numpy(dtype=float, na_value=np.nan)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Split by sign so neither exp() branch overflows.
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclass(frozen=True)
class LinearScorer:
    """Logistic regression as coefficients on already-encoded features.

    Any standardisation applied during fitting is folded into the coefficients on
    export, so scoring is a single dot product and needs no scaler object.
    """

    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    kind: str = "linear"

    def __post_init__(self) -> None:
        if len(self.features) != len(self.coefficients):
            raise ValueError("features and coefficients must have the same length")

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        values = _matrix(X, self.features)
        if np.isnan(values).any():
            # WOE encoding leaves no gaps; a NaN here means an unencoded column.
            raise ValueError("linear scorer received NaN inputs; encode features first")
        return values @ np.asarray(self.coefficients, dtype=float) + self.intercept

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.margin(X))

    def coefficient_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": list(self.features), "coefficient": list(self.coefficients)}
        ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            "kind": "linear",
            "features": list(self.features),
            "coefficients": [float(c) for c in self.coefficients],
            "intercept": float(self.intercept),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "LinearScorer":
        return cls(
            features=tuple(data["features"]),
            coefficients=tuple(float(c) for c in data["coefficients"]),
            intercept=float(data["intercept"]),
        )


@dataclass(frozen=True)
class Tree:
    """One decision tree flattened into parallel arrays.

    `feature[i] < 0` marks a leaf. `missing[i]` is the child a NaN takes, which
    is what lets the ensemble keep the trainer's learned missing-value direction
    instead of guessing at scoring time.
    """

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    missing: np.ndarray
    value: np.ndarray

    def to_dict(self) -> dict:
        return {
            "feature": self.feature.astype(int).tolist(),
            "threshold": self.threshold.astype(float).tolist(),
            "left": self.left.astype(int).tolist(),
            "right": self.right.astype(int).tolist(),
            "missing": self.missing.astype(int).tolist(),
            "value": self.value.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Tree":
        return cls(
            feature=np.asarray(data["feature"], dtype=int),
            threshold=np.asarray(data["threshold"], dtype=float),
            left=np.asarray(data["left"], dtype=int),
            right=np.asarray(data["right"], dtype=int),
            missing=np.asarray(data["missing"], dtype=int),
            value=np.asarray(data["value"], dtype=float),
        )


@dataclass(frozen=True)
class TreeEnsemble:
    """Additive tree ensemble in log-odds space.

    Two details are recorded rather than assumed, because both only bite on rows
    sitting exactly on a split — the quietest possible form of scoring drift:

    `split_op` is the trainer's comparison convention (XGBoost sends
    `x < threshold` left, scikit-learn sends `x <= threshold` left), and `dtype`
    is the precision it compared in (XGBoost casts inputs to float32).
    """

    features: tuple[str, ...]
    trees: tuple[Tree, ...]
    base_margin: float
    split_op: str = "lt"
    dtype: str = "float64"
    kind: str = "tree_ensemble"

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        values = _matrix(X, self.features)
        if self.dtype == "float32":
            values = values.astype(np.float32)
        total = np.full(len(values), self.base_margin, dtype=float)
        for tree in self.trees:
            total += _apply_tree(tree, values, self.split_op, self.dtype)
        return total

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.margin(X))

    def to_dict(self) -> dict:
        return {
            "kind": "tree_ensemble",
            "features": list(self.features),
            "base_margin": float(self.base_margin),
            "split_op": self.split_op,
            "dtype": self.dtype,
            "trees": [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "TreeEnsemble":
        return cls(
            features=tuple(data["features"]),
            trees=tuple(Tree.from_dict(t) for t in data["trees"]),
            base_margin=float(data["base_margin"]),
            split_op=data.get("split_op", "lt"),
            dtype=data.get("dtype", "float64"),
        )


def _apply_tree(tree: Tree, values: np.ndarray, split_op: str, dtype: str = "float64") -> np.ndarray:
    """Walk every row down one tree at once, one depth level per iteration."""
    n_rows = len(values)
    node = np.zeros(n_rows, dtype=int)
    cast = np.float32 if dtype == "float32" else np.float64
    # Depth is bounded by the node count; the guard only exists to make a
    # corrupted tree fail loudly instead of hanging.
    for _ in range(len(tree.feature) + 1):
        internal = tree.feature[node] >= 0
        if not internal.any():
            return tree.value[node]
        rows = np.nonzero(internal)[0]
        here = node[rows]
        x = values[rows, tree.feature[here]]
        threshold = tree.threshold[here].astype(cast)
        go_left = x < threshold if split_op == "lt" else x <= threshold
        nxt = np.where(go_left, tree.left[here], tree.right[here])
        node[rows] = np.where(np.isnan(x), tree.missing[here], nxt)
    raise RuntimeError("tree traversal did not terminate; the ensemble is malformed")


@dataclass(frozen=True)
class IsotonicCurve:
    """Monotone probability calibration stored as an interpolation table.

    Maps a raw model score to the bad rate actually observed at that score,
    clipped at the ends of the fitted range.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]

    def predict(self, scores) -> np.ndarray:
        values = np.asarray(pd.Series(scores).to_numpy(), dtype=float)
        if not self.x:
            return values
        return np.interp(values, np.asarray(self.x), np.asarray(self.y))

    def to_dict(self) -> dict:
        return {"x": [float(v) for v in self.x], "y": [float(v) for v in self.y]}

    @classmethod
    def from_dict(cls, data: Mapping) -> "IsotonicCurve":
        return cls(x=tuple(float(v) for v in data["x"]), y=tuple(float(v) for v in data["y"]))


def predictor_from_dict(data: Mapping) -> Predictor:
    kind = data.get("kind")
    if kind == "linear":
        return LinearScorer.from_dict(data)
    if kind == "tree_ensemble":
        return TreeEnsemble.from_dict(data)
    raise ValueError(f"unknown predictor kind '{kind}'")
