"""The scoring bundle: everything needed to turn an application into a decision.

A bundle is one JSON file. It carries the schema, the binnings, the feature
recipe, the model, the calibration curve, the decision policy and the drift
baseline — and nothing that requires a specific library version to unpickle. The
consequence is that scoring a batch needs numpy and pandas and nothing else, and
that a bundle from a year ago still loads today.

`ScoringBundle.score()` is the *only* scoring path in the codebase. Training
calls it to report what a run would have decided, tests call it to prove train
and serve agree, and the CLI calls it in production. There is no second
implementation to drift away from it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data.schema import DatasetSchema, validate_frame
from .features.space import FeatureSpace
from .features.woe import WoeTransformer
from .logging_setup import get_logger
from .models.predictors import IsotonicCurve, Predictor, predictor_from_dict
from .models.scorecard import to_credit_score
from .monitoring.drift import DriftBaseline
from .policy.decision import DecisionPolicy
from .settings import ScorecardSettings

log = get_logger("bundle")

FORMAT_VERSION = 1
BUNDLE_FILENAME = "bundle.json"


@dataclass(frozen=True)
class ScoringBundle:
    schema: DatasetSchema
    woe: WoeTransformer
    space: FeatureSpace
    predictor: Predictor
    policy: DecisionPolicy
    drift: DriftBaseline
    calibrator: IsotonicCurve | None = None
    scorecard_settings: ScorecardSettings = field(default_factory=ScorecardSettings)
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- scoring

    def raw_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.predictor.predict_proba(self.space.build(df, self.woe))

    def score(self, df: pd.DataFrame, include_decisions: bool = True) -> pd.DataFrame:
        """Score a frame and apply the policy.

        Returns the model's raw probability, the calibrated expected bad rate,
        the credit score, and — unless suppressed — the decision with its reason.
        """
        warnings = validate_frame(df, self.schema, require_label=False)
        for message in warnings:
            log.warning("input check: %s", message)

        raw = self.raw_scores(df)
        out = pd.DataFrame(index=df.index)
        if self.schema.id_col and self.schema.id_col in df.columns:
            out[self.schema.id_col] = df[self.schema.id_col].to_numpy()
        out["model_score"] = raw
        out["calibrated_prob"] = self.calibrator.predict(raw) if self.calibrator else raw
        out["credit_score"] = np.round(to_credit_score(raw, self.scorecard_settings), 1)
        if include_decisions:
            out = pd.concat([out, self.policy.decide(df, raw)], axis=1)
        return out

    def drift_report(self, df: pd.DataFrame, settings) -> pd.DataFrame:
        return self.drift.report(df, self.woe, settings, scores=self.raw_scores(df))

    def with_policy(self, policy: DecisionPolicy) -> "ScoringBundle":
        """A copy carrying a different policy — for what-if analysis on the same model."""
        return replace(self, policy=policy)

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "schema": self.schema.to_dict(),
            "woe": self.woe.to_dict(),
            "space": self.space.to_dict(),
            "predictor": self.predictor.to_dict(),
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
            "policy": self.policy.to_dict(),
            "drift": self.drift.to_dict(),
            "scorecard": {
                "pdo": self.scorecard_settings.pdo,
                "base_score": self.scorecard_settings.base_score,
                "base_odds": self.scorecard_settings.base_odds,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ScoringBundle":
        version = int(data.get("format_version", 0))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"bundle format version {version} cannot be read by this release "
                f"(expected {FORMAT_VERSION})"
            )
        scorecard = data.get("scorecard", {})
        calibrator = data.get("calibrator")
        return cls(
            schema=DatasetSchema.from_dict(data["schema"]),
            woe=WoeTransformer.from_dict(data["woe"]),
            space=FeatureSpace.from_dict(data["space"]),
            predictor=predictor_from_dict(data["predictor"]),
            policy=DecisionPolicy.from_dict(data["policy"]),
            drift=DriftBaseline.from_dict(data.get("drift", {})),
            calibrator=IsotonicCurve.from_dict(calibrator) if calibrator else None,
            scorecard_settings=ScorecardSettings(
                pdo=float(scorecard.get("pdo", 20.0)),
                base_score=float(scorecard.get("base_score", 600.0)),
                base_odds=float(scorecard.get("base_odds", 1 / 15)),
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.is_dir():
            target = target / BUNDLE_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        log.info("saved scoring bundle to %s (%.1f KB)", target, target.stat().st_size / 1024)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ScoringBundle":
        source = Path(path)
        if source.is_dir():
            source = source / BUNDLE_FILENAME
        if not source.exists():
            raise FileNotFoundError(f"no scoring bundle at {source}")
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8-sig")))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)
