"""Supervised binning and weight of evidence.

One binning implementation serves every consumer — feature screening, the
scorecard, the GBDT's categorical handling and the rule miner — so a feature's
cut points are identical everywhere. A fitted binning is a frozen dataclass that
round-trips through JSON and evaluates with numpy alone, which is what lets the
scoring service run without scikit-learn installed.

Convention: WOE = ln(bad_share / good_share), so a **higher WOE means higher
risk**. Model coefficients, monotone constraints and score direction all follow
from that single choice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..settings import BinningSettings

MISSING_BIN = -1
OTHER_LEVEL = "__other__"
MISSING_LEVEL = "__missing__"
# Continuity correction: a bin with zero bads still gets a finite WOE.
_SMOOTHING = 0.5


class BinningError(ValueError):
    """A feature cannot be binned (constant, all-missing, too few levels)."""


def canonical(value) -> str | None:
    """Map one categorical value to the key it is stored under. None for nulls.

    Plain `str()` is not enough. A column of small integers is read back as
    float64 the moment any row is missing, so a level stored as ``"12"`` during
    training arrives as ``"12.0"`` at scoring time and silently becomes an
    unseen category. Collapsing integral floats onto the integer spelling makes
    the key survive that round trip, which is the difference between a level
    keeping its fitted WOE and quietly falling through to the pooled bucket.
    """
    if value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    return str(value)


def as_text(values) -> list[str | None]:
    """Normalise a categorical column to canonical keys, nulls as None.

    pandas has several null sentinels (None, NaN, NA, NaT) and which one survives
    a given operation varies by version and dtype; collapsing them here keeps the
    rest of the codebase from having to care.
    """
    return [canonical(v) for v in pd.Series(values).tolist()]


@dataclass(frozen=True)
class BinStat:
    index: int
    label: str
    rows: int
    bads: int
    bad_rate: float
    woe: float
    iv: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NumericBinning:
    """Interval binning of a numeric feature.

    `cuts` are the interior right-open boundaries: bin i covers
    (cuts[i-1], cuts[i]], with bin 0 unbounded below and the last unbounded above.
    """

    feature: str
    cuts: tuple[float, ...]
    woe: tuple[float, ...]
    missing_woe: float
    missing_own_bin: bool
    stats: tuple[BinStat, ...]
    iv: float
    kind: str = "numeric"

    def assign(self, values: Sequence | pd.Series | np.ndarray) -> np.ndarray:
        """Bin index per row; MISSING_BIN for nulls."""
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        out = np.full(len(arr), MISSING_BIN, dtype=int)
        present = ~np.isnan(arr)
        if self.cuts:
            # side="left" makes the interval right-closed: x <= cut stays left.
            out[present] = np.searchsorted(np.asarray(self.cuts, dtype=float), arr[present], side="left")
        else:
            out[present] = 0
        return out

    def transform(self, values) -> np.ndarray:
        idx = self.assign(values)
        table = np.asarray(self.woe, dtype=float)
        out = np.where(idx == MISSING_BIN, self.missing_woe, table[np.clip(idx, 0, len(table) - 1)])
        return out.astype(float)

    def bin_labels(self) -> list[str]:
        return [s.label for s in self.stats]

    def to_dict(self) -> dict:
        return {
            "kind": "numeric",
            "feature": self.feature,
            "cuts": list(self.cuts),
            "woe": list(self.woe),
            "missing_woe": self.missing_woe,
            "missing_own_bin": self.missing_own_bin,
            "iv": self.iv,
            "stats": [s.to_dict() for s in self.stats],
        }


@dataclass(frozen=True)
class CategoricalBinning:
    """Level-to-WOE mapping, with rare and unseen levels pooled together."""

    feature: str
    levels: tuple[str, ...]
    woe: tuple[float, ...]
    other_woe: float
    missing_woe: float
    stats: tuple[BinStat, ...]
    iv: float
    kind: str = "categorical"

    def _lookup(self) -> dict[str, float]:
        return dict(zip(self.levels, self.woe))

    def assign(self, values) -> np.ndarray:
        index = {level: i for i, level in enumerate(self.levels)}
        fallback = index.get(OTHER_LEVEL, MISSING_BIN)
        return np.array(
            [MISSING_BIN if v is None else index.get(v, fallback) for v in as_text(values)],
            dtype=int,
        )

    def transform(self, values) -> np.ndarray:
        lookup = self._lookup()
        return np.array(
            [
                self.missing_woe if v is None else lookup.get(v, self.other_woe)
                for v in as_text(values)
            ],
            dtype=float,
        )

    def bin_labels(self) -> list[str]:
        return [s.label for s in self.stats]

    def to_dict(self) -> dict:
        return {
            "kind": "categorical",
            "feature": self.feature,
            "levels": list(self.levels),
            "woe": list(self.woe),
            "other_woe": self.other_woe,
            "missing_woe": self.missing_woe,
            "iv": self.iv,
            "stats": [s.to_dict() for s in self.stats],
        }


Binning = NumericBinning | CategoricalBinning


def binning_from_dict(data: Mapping) -> Binning:
    stats = tuple(BinStat(**s) for s in data.get("stats", []))
    if data["kind"] == "numeric":
        return NumericBinning(
            feature=data["feature"],
            cuts=tuple(float(c) for c in data["cuts"]),
            woe=tuple(float(w) for w in data["woe"]),
            missing_woe=float(data["missing_woe"]),
            missing_own_bin=bool(data["missing_own_bin"]),
            stats=stats,
            iv=float(data["iv"]),
        )
    return CategoricalBinning(
        feature=data["feature"],
        levels=tuple(str(v) for v in data["levels"]),
        woe=tuple(float(w) for w in data["woe"]),
        other_woe=float(data["other_woe"]),
        missing_woe=float(data["missing_woe"]),
        stats=stats,
        iv=float(data["iv"]),
    )


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def fit_numeric(
    values,
    y,
    settings: BinningSettings,
    feature: str = "feature",
) -> NumericBinning:
    x = pd.to_numeric(pd.Series(values).reset_index(drop=True), errors="coerce")
    target = pd.Series(y).reset_index(drop=True).astype(float)
    if len(x) != len(target):
        raise BinningError(f"{feature}: values and labels differ in length")
    _require_both_classes(target, feature)

    present = x.notna().to_numpy()
    x_present = x.to_numpy(dtype=float)[present]
    y_present = target.to_numpy()[present]
    if x_present.size == 0:
        raise BinningError(f"{feature}: every value is missing")
    if np.unique(x_present).size < 2:
        raise BinningError(f"{feature}: constant across non-missing rows")

    cuts = _candidate_cuts(x_present, y_present, settings)
    if settings.enforce_monotonic:
        cuts = _merge_to_monotone(x_present, y_present, cuts, settings)
    cuts = _merge_small_bins(x_present, cuts, settings)

    bin_index = _numeric_bins(x_present, cuts)
    n_bins = len(cuts) + 1
    counts, bads = _bin_counts(bin_index, y_present, n_bins)

    missing_rows = int((~present).sum())
    missing_bads = float(target.to_numpy()[~present].sum()) if missing_rows else 0.0
    missing_own_bin = missing_rows >= settings.missing_min_rows

    all_counts = np.append(counts, missing_rows) if missing_own_bin else counts
    all_bads = np.append(bads, missing_bads) if missing_own_bin else bads
    woe, iv_parts = _woe_from_counts(all_counts, all_bads)

    labels = _interval_labels(cuts)
    if missing_own_bin:
        labels = labels + [MISSING_LEVEL]

    stats = tuple(
        BinStat(
            index=(MISSING_BIN if (missing_own_bin and i == len(labels) - 1) else i),
            label=label,
            rows=int(all_counts[i]),
            bads=int(all_bads[i]),
            bad_rate=round(float(all_bads[i] / all_counts[i]) if all_counts[i] else 0.0, 6),
            woe=round(float(woe[i]), 6),
            iv=round(float(iv_parts[i]), 6),
        )
        for i, label in enumerate(labels)
    )
    return NumericBinning(
        feature=feature,
        cuts=tuple(float(c) for c in cuts),
        woe=tuple(round(float(w), 6) for w in woe[:n_bins]),
        missing_woe=round(float(woe[-1]), 6) if missing_own_bin else 0.0,
        missing_own_bin=missing_own_bin,
        stats=stats,
        iv=round(float(iv_parts.sum()), 6),
    )


def fit_categorical(
    values,
    y,
    settings: BinningSettings,
    feature: str = "feature",
) -> CategoricalBinning:
    series = pd.Series(values).reset_index(drop=True)
    target = pd.Series(y).reset_index(drop=True).astype(float)
    if len(series) != len(target):
        raise BinningError(f"{feature}: values and labels differ in length")
    _require_both_classes(target, feature)

    text = as_text(series)
    present = np.array([v is not None for v in text])
    if present.sum() == 0:
        raise BinningError(f"{feature}: every value is missing")

    frequency = pd.Series([v for v in text if v is not None]).value_counts()
    min_rows = max(1, int(settings.rare_category_rate * len(text)))
    common = {level for level, count in frequency.items() if count >= min_rows}
    if not common:
        common = {str(frequency.index[0])}
    # Levels too rare to estimate a reliable WOE for share one pooled bin, which
    # also gives unseen levels somewhere sane to land at scoring time.
    pooled = np.array(
        [None if v is None else (v if v in common else OTHER_LEVEL) for v in text], dtype=object
    )

    levels = sorted({v for v in pooled if v is not None})
    if len(levels) < 2:
        raise BinningError(f"{feature}: fewer than 2 usable levels after pooling")

    y_values = target.to_numpy()
    counts = np.array([float((pooled == level).sum()) for level in levels])
    bads = np.array([float(y_values[pooled == level].sum()) for level in levels])

    missing_rows = int((~present).sum())
    missing_own_bin = missing_rows >= settings.missing_min_rows
    if missing_own_bin:
        counts = np.append(counts, missing_rows)
        bads = np.append(bads, float(target.to_numpy()[~present].sum()))

    woe, iv_parts = _woe_from_counts(counts, bads)
    labels = list(levels) + ([MISSING_LEVEL] if missing_own_bin else [])
    stats = tuple(
        BinStat(
            index=(MISSING_BIN if (missing_own_bin and i == len(labels) - 1) else i),
            label=label,
            rows=int(counts[i]),
            bads=int(bads[i]),
            bad_rate=round(float(bads[i] / counts[i]) if counts[i] else 0.0, 6),
            woe=round(float(woe[i]), 6),
            iv=round(float(iv_parts[i]), 6),
        )
        for i, label in enumerate(labels)
    )
    level_woe = tuple(round(float(w), 6) for w in woe[: len(levels)])
    # Unseen levels at scoring time inherit the pooled "other" WOE when one was
    # fitted, and a neutral 0 otherwise.
    other_woe = level_woe[levels.index(OTHER_LEVEL)] if OTHER_LEVEL in levels else 0.0
    return CategoricalBinning(
        feature=feature,
        levels=tuple(levels),
        woe=level_woe,
        other_woe=other_woe,
        missing_woe=round(float(woe[-1]), 6) if missing_own_bin else 0.0,
        stats=stats,
        iv=round(float(iv_parts.sum()), 6),
    )


def fit_binning(values, y, kind: str, settings: BinningSettings, feature: str) -> Binning:
    if kind == "numeric":
        return fit_numeric(values, y, settings, feature)
    if kind == "categorical":
        return fit_categorical(values, y, settings, feature)
    raise ValueError(f"unknown feature kind '{kind}'")


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _require_both_classes(target: pd.Series, feature: str) -> None:
    """WOE is undefined against a single class.

    With no bads (or no goods) the smoothing floor is the only thing keeping the
    log finite, and every bin ends up with the same fabricated share — which
    yields an enormous, entirely meaningless IV. Refusing is the only honest
    answer.
    """
    if target.dropna().nunique() < 2:
        raise BinningError(
            f"{feature}: the label has a single class, so weight of evidence is undefined"
        )


def _candidate_cuts(x: np.ndarray, y: np.ndarray, settings: BinningSettings) -> list[float]:
    """Cut points from a shallow decision tree, falling back to quantiles."""
    min_leaf = max(settings.min_bin_rows, int(settings.min_bin_fraction * len(x)), 1)
    try:
        from sklearn.tree import DecisionTreeClassifier

        tree = DecisionTreeClassifier(
            max_leaf_nodes=max(2, settings.max_bins),
            min_samples_leaf=min_leaf,
            random_state=settings.random_state,
        )
        tree.fit(x.reshape(-1, 1), y)
        raw = tree.tree_.threshold[tree.tree_.feature >= 0]
        cuts = sorted({float(t) for t in raw})
    except Exception:
        cuts = []
    if not cuts:
        quantiles = np.linspace(0, 1, max(2, settings.max_bins) + 1)[1:-1]
        cuts = sorted({float(v) for v in np.quantile(x, quantiles)})
        cuts = [c for c in cuts if c < x.max()]
    return cuts


def _numeric_bins(x: np.ndarray, cuts: Sequence[float]) -> np.ndarray:
    if len(cuts) == 0:
        return np.zeros(len(x), dtype=int)
    return np.searchsorted(np.asarray(cuts, dtype=float), x, side="left")


def _bin_counts(bin_index: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(bin_index, minlength=n_bins).astype(float)
    bads = np.bincount(bin_index, weights=y, minlength=n_bins).astype(float)
    return counts[:n_bins], bads[:n_bins]


def _woe_from_counts(counts: np.ndarray, bads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WOE = ln(bad share / good share) with a 0.5 continuity correction."""
    goods = counts - bads
    total_bad = max(bads.sum(), _SMOOTHING)
    total_good = max(goods.sum(), _SMOOTHING)
    bad_share = np.maximum(bads, _SMOOTHING) / total_bad
    good_share = np.maximum(goods, _SMOOTHING) / total_good
    woe = np.log(bad_share / good_share)
    iv = (bad_share - good_share) * woe
    return woe, iv


def _merge_to_monotone(
    x: np.ndarray, y: np.ndarray, cuts: list[float], settings: BinningSettings
) -> list[float]:
    """Drop cut points until the bad rate is monotone across bins.

    The target direction is whichever of increasing/decreasing the raw bin rates
    already lean toward; each step removes the single cut whose two neighbouring
    bins violate that direction by the least, so the strongest structure
    survives.
    """
    cuts = list(cuts)
    while cuts:
        counts, bads = _bin_counts(_numeric_bins(x, cuts), y, len(cuts) + 1)
        if (counts == 0).any():
            cuts.pop(min(int(np.argmax(counts == 0)), len(cuts) - 1))
            continue
        if len(counts) < 3:
            break
        rates = bads / counts
        increasing = _direction(rates)
        diffs = np.diff(rates)
        violations = diffs < 0 if increasing else diffs > 0
        if not violations.any():
            break
        # diffs[i] compares bin i with bin i+1, so cuts[i] is the boundary to drop.
        offending = np.where(violations)[0]
        cuts.pop(int(offending[int(np.argmin(np.abs(diffs[offending])))]))
    return cuts


def _direction(rates: np.ndarray) -> bool:
    """True when bad rate broadly increases with the bin index."""
    if len(rates) < 2:
        return True
    ranks = np.arange(len(rates), dtype=float)
    centered_x = ranks - ranks.mean()
    centered_y = rates - rates.mean()
    slope = float((centered_x * centered_y).sum())
    return slope >= 0


def _merge_small_bins(x: np.ndarray, cuts: list[float], settings: BinningSettings) -> list[float]:
    """Remove cuts that leave a bin below the minimum row count."""
    min_rows = max(settings.min_bin_rows, int(settings.min_bin_fraction * len(x)), 1)
    cuts = list(cuts)
    while cuts:
        counts = np.bincount(_numeric_bins(x, cuts), minlength=len(cuts) + 1)
        if counts.min() >= min_rows or len(counts) <= 2:
            break
        smallest = int(np.argmin(counts))
        # Merging bin i means dropping the boundary that separates it from the
        # neighbour it is closest in size to.
        if smallest == 0:
            cuts.pop(0)
        elif smallest == len(counts) - 1:
            cuts.pop(-1)
        else:
            left, right = counts[smallest - 1], counts[smallest + 1]
            cuts.pop(smallest - 1 if left <= right else smallest)
    return cuts


def _interval_labels(cuts: Sequence[float]) -> list[str]:
    edges = [-np.inf, *cuts, np.inf]
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_text = "-inf" if np.isneginf(lo) else f"{lo:.6g}"
        hi_text = "inf" if np.isposinf(hi) else f"{hi:.6g}"
        labels.append(f"({lo_text}, {hi_text}]")
    return labels


def bin_table(binnings: Iterable[Binning]) -> pd.DataFrame:
    """Flatten fitted binnings into one reportable table."""
    rows = []
    for binning in binnings:
        for stat in binning.stats:
            rows.append(
                {
                    "feature": binning.feature,
                    "kind": binning.kind,
                    "bin": stat.index,
                    "label": stat.label,
                    "rows": stat.rows,
                    "bads": stat.bads,
                    "bad_rate": stat.bad_rate,
                    "woe": stat.woe,
                    "bin_iv": stat.iv,
                    "feature_iv": binning.iv,
                }
            )
    return pd.DataFrame(rows)
