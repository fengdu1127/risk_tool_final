"""Population drift between the training sample and a scored batch.

The baseline is expressed in the model's *own* bins rather than in fresh
quantiles of the new batch. That matters: re-quantiling each batch measures its
internal shape and can report a perfectly stable population while the model's
inputs have shifted underneath it. Comparing bin occupancy answers the question
that actually matters — are applicants still landing where the model learned
they would?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..features.woe import WoeTransformer
from ..logging_setup import get_logger
from ..settings import MonitoringSettings

log = get_logger("drift")

_FLOOR = 1e-6


@dataclass(frozen=True)
class DriftBaseline:
    """Training-time bin occupancy per feature, plus the score distribution."""

    shares: dict[str, dict[str, float]]
    missing_rate: dict[str, float]
    score_edges: tuple[float, ...]
    score_shares: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        woe: WoeTransformer,
        features: Sequence[str],
        scores: np.ndarray | None = None,
        n_score_bands: int = 10,
    ) -> "DriftBaseline":
        shares: dict[str, dict[str, float]] = {}
        missing: dict[str, float] = {}
        for feature in features:
            binning = woe.binnings.get(feature)
            if binning is None or feature not in df.columns:
                continue
            assigned = binning.assign(df[feature])
            shares[feature] = _shares(assigned)
            missing[feature] = round(float(df[feature].isna().mean()), 6)

        score_edges: tuple[float, ...] = ()
        score_shares: tuple[float, ...] = ()
        if scores is not None and len(scores):
            edges = np.unique(np.quantile(np.asarray(scores, dtype=float), np.linspace(0, 1, n_score_bands + 1)[1:-1]))
            bands = np.clip(np.searchsorted(edges, scores, side="left"), 0, len(edges))
            counts = np.bincount(bands, minlength=len(edges) + 1).astype(float)
            score_edges = tuple(float(e) for e in edges)
            score_shares = tuple(float(v) for v in counts / counts.sum())

        return cls(shares=shares, missing_rate=missing, score_edges=score_edges, score_shares=score_shares)

    def report(
        self,
        df: pd.DataFrame,
        woe: WoeTransformer,
        settings: MonitoringSettings,
        scores: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Per-feature PSI and missing-rate shift for a new batch."""
        rows = []
        for feature, expected in self.shares.items():
            binning = woe.binnings.get(feature)
            if binning is None or feature not in df.columns:
                rows.append({"feature": feature, "psi": np.nan, "level": "absent", "note": "column not in batch"})
                continue
            actual = _shares(binning.assign(df[feature]))
            value = _psi(expected, actual)
            new_missing = float(df[feature].isna().mean())
            reference_missing = self.missing_rate.get(feature, 0.0)
            rows.append(
                {
                    "feature": feature,
                    "psi": round(value, 6),
                    "level": _level(value, settings),
                    "train_missing_rate": round(reference_missing, 6),
                    "batch_missing_rate": round(new_missing, 6),
                    "missing_rate_shift": round(new_missing - reference_missing, 6),
                    "note": "",
                }
            )

        if scores is not None and self.score_shares:
            edges = np.asarray(self.score_edges, dtype=float)
            bands = np.clip(np.searchsorted(edges, np.asarray(scores, dtype=float), side="left"), 0, len(edges))
            counts = np.bincount(bands, minlength=len(self.score_shares)).astype(float)
            actual = {str(i): v for i, v in enumerate(counts / max(counts.sum(), 1))}
            expected = {str(i): v for i, v in enumerate(self.score_shares)}
            value = _psi(expected, actual)
            rows.append(
                {
                    "feature": "__model_score__",
                    "psi": round(value, 6),
                    "level": _level(value, settings),
                    "note": "distribution of the model score itself",
                }
            )

        report = pd.DataFrame(rows)
        return report.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True) if len(report) else report

    def to_dict(self) -> dict:
        return {
            "shares": self.shares,
            "missing_rate": self.missing_rate,
            "score_edges": list(self.score_edges),
            "score_shares": list(self.score_shares),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "DriftBaseline":
        return cls(
            shares={f: {str(k): float(v) for k, v in s.items()} for f, s in data.get("shares", {}).items()},
            missing_rate={f: float(v) for f, v in data.get("missing_rate", {}).items()},
            score_edges=tuple(float(v) for v in data.get("score_edges", [])),
            score_shares=tuple(float(v) for v in data.get("score_shares", [])),
        )


def alerts(report: pd.DataFrame, settings: MonitoringSettings) -> list[str]:
    """Human-readable warnings worth putting in front of an operator."""
    if report.empty:
        return []
    messages = []
    unstable = report.loc[report["level"] == "alert", "feature"].tolist()
    warned = report.loc[report["level"] == "warn", "feature"].tolist()
    if unstable:
        messages.append(f"PSI at or above {settings.psi_alert}: {', '.join(unstable)}")
    if warned:
        messages.append(f"PSI between {settings.psi_warn} and {settings.psi_alert}: {', '.join(warned)}")
    if "missing_rate_shift" in report.columns:
        shifted = report.loc[
            report["missing_rate_shift"].abs() > settings.missing_rate_shift_alert, "feature"
        ].tolist()
        if shifted:
            messages.append(
                f"missing rate moved more than {settings.missing_rate_shift_alert:.0%}: {', '.join(shifted)}"
            )
    return messages


def _shares(assigned: np.ndarray) -> dict[str, float]:
    if len(assigned) == 0:
        return {}
    values, counts = np.unique(assigned, return_counts=True)
    total = counts.sum()
    return {str(int(v)): round(float(c / total), 6) for v, c in zip(values, counts)}


def _psi(expected: Mapping[str, float], actual: Mapping[str, float]) -> float:
    keys = sorted(set(expected) | set(actual))
    if not keys:
        return float("nan")
    e = np.array([max(expected.get(k, 0.0), _FLOOR) for k in keys])
    a = np.array([max(actual.get(k, 0.0), _FLOOR) for k in keys])
    return float(np.sum((a - e) * np.log(a / e)))


def _level(value: float, settings: MonitoringSettings) -> str:
    if np.isnan(value):
        return "unknown"
    if value >= settings.psi_alert:
        return "alert"
    if value >= settings.psi_warn:
        return "warn"
    return "stable"
