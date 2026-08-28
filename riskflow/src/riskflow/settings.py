"""Typed, immutable configuration.

The whole config tree is frozen dataclasses rather than module-level dicts, so a
run cannot mutate global state that a later run then inherits. Overrides are
applied functionally (`Settings.merged(...)`) and every run snapshots the
settings it actually used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SplitSettings:
    """How raw rows become train / test / holdout."""

    train_ratio: float = 0.70
    test_ratio: float = 0.15
    holdout_ratio: float = 0.15
    random_state: int = 42
    # When set, the newest `oot_months` of data becomes an out-of-time holdout
    # and everything older is split randomly into train/test.
    time_col: str | None = None
    oot_months: int = 3
    min_holdout_rows: int = 50


@dataclass(frozen=True)
class BinningSettings:
    """Supervised binning used by every WOE consumer."""

    max_bins: int = 8
    min_bin_rows: int = 50
    min_bin_fraction: float = 0.03
    # Missing values become their own bin once train has at least this many;
    # below that they take a neutral WOE of 0.
    missing_min_rows: int = 50
    # Merge adjacent bins until bad rate is monotone in the bin order.
    enforce_monotonic: bool = True
    # Categories rarer than this are pooled into an "__other__" level.
    rare_category_rate: float = 0.01
    random_state: int = 42


@dataclass(frozen=True)
class ScreeningSettings:
    """Gates a feature must pass to reach the model / rule miner."""

    max_missing_rate: float = 0.50
    min_iv: float = 0.02
    # An implausibly high IV usually means label leakage, not a great feature.
    max_iv: float = 1.50
    max_psi: float = 0.10
    max_abs_corr: float = 0.70
    max_vif: float = 10.0
    min_monotonic_corr: float = 0.60


@dataclass(frozen=True)
class ScorecardSettings:
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 1.0 / 15.0


@dataclass(frozen=True)
class ModelSettings:
    algorithms: tuple[str, ...] = ("logistic", "gbdt")
    primary_metric: str = "ks"
    search_iterations: int = 24
    cv_folds: int = 4
    random_state: int = 42
    balance_classes: bool = True
    # Force GBDT splits to respect each feature's risk direction.
    monotone_constraints: bool = True
    monotone_min_abs_corr: float = 0.02
    calibration: str = "isotonic"  # "isotonic" | "none"
    scorecard: ScorecardSettings = field(default_factory=ScorecardSettings)


@dataclass(frozen=True)
class RuleSettings:
    """Constraints on a single hard-reject rule."""

    max_coverage: float = 0.05
    min_lift: float = 2.0
    min_hits: int = 5
    # Lift may not decay more than this from test to holdout.
    max_lift_decay: float = 0.35
    # Two rules hitting mostly the same rows: keep the stronger one.
    max_overlap: float = 0.80
    max_rules: int = 30
    # Candidate thresholds are placed by target coverage rather than on an even
    # quantile grid: a rule that must cover under `max_coverage` is only ever
    # found in the tail, so that is where the grid needs its resolution.
    grid_points: int = 14
    tree_max_depth: int = 3
    tree_min_leaf_rows: int = 50
    # Several trees over different random feature subsets, for a candidate pool
    # that is not dominated by whichever feature splits best first.
    tree_count: int = 6
    tree_feature_sample: int = 4
    random_state: int = 42


@dataclass(frozen=True)
class CutoffSettings:
    """Score-threshold search for reject / review / approve."""

    reject_rate_grid: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10)
    review_rate_grid: tuple[float, ...] = (0.05, 0.10, 0.15)
    # Reject as widely as possible while the rejected pool stays this much worse
    # than the book; same idea for the softer review band.
    min_reject_lift: float = 2.5
    min_review_lift: float = 1.2
    segment_features: tuple[str, ...] = ()
    segment_min_rows: int = 200
    segment_min_holdout_hits: int = 10
    segment_min_lift: float = 2.0
    segment_max_lift_gap: float = 0.50
    segment_min_share: float = 0.05
    segment_min_bad_rate_gap: float = 0.02


@dataclass(frozen=True)
class MonitoringSettings:
    psi_bins: int = 10
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    missing_rate_shift_alert: float = 0.10


@dataclass(frozen=True)
class ReportSettings:
    output_root: str = "runs"
    make_plots: bool = True
    fig_dpi: int = 120
    save_scored_rows: bool = False


@dataclass(frozen=True)
class Settings:
    split: SplitSettings = field(default_factory=SplitSettings)
    binning: BinningSettings = field(default_factory=BinningSettings)
    screening: ScreeningSettings = field(default_factory=ScreeningSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    rules: RuleSettings = field(default_factory=RuleSettings)
    cutoffs: CutoffSettings = field(default_factory=CutoffSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    report: ReportSettings = field(default_factory=ReportSettings)

    def merged(self, overrides: Mapping[str, Any] | None) -> "Settings":
        """Return a new Settings with `overrides` applied. Never mutates self."""
        if not overrides:
            return self
        return _apply(self, overrides)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Settings":
        return cls().merged(data)

    @classmethod
    def load(cls, path: str | Path | None) -> "Settings":
        """Build settings from an optional JSON override file."""
        if path is None:
            return cls()
        # utf-8-sig tolerates the BOM Windows editors prepend.
        raw = Path(path).read_text(encoding="utf-8-sig")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object at the top level")
        return cls.from_dict(data)


def _apply(obj: T, overrides: Mapping[str, Any]) -> T:
    """Recursively override dataclass fields, validating names and coercing types."""
    known = {f.name: f for f in fields(obj)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in known:
            raise KeyError(
                f"unknown setting '{key}' for {type(obj).__name__}; "
                f"valid keys: {sorted(known)}"
            )
        current = getattr(obj, key)
        if is_dataclass(current) and not isinstance(current, type):
            if not isinstance(value, Mapping):
                raise TypeError(f"setting '{key}' expects an object, got {type(value).__name__}")
            changes[key] = _apply(current, value)
        else:
            changes[key] = _coerce(value, current, key)
    return replace(obj, **changes)


def _coerce(value: Any, current: Any, key: str) -> Any:
    """Keep tuple-typed settings tuples so they stay hashable and immutable."""
    if isinstance(current, tuple):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise TypeError(f"setting '{key}' expects a list, got {type(value).__name__}")
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"setting '{key}' expects a boolean, got {type(value).__name__}")
        return value
    if isinstance(current, int) and not isinstance(current, bool) and isinstance(value, bool):
        raise TypeError(f"setting '{key}' expects a number, got bool")
    return value
