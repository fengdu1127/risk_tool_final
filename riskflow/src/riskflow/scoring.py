"""Batch scoring against a promoted run.

Scoring deliberately does two things beyond producing numbers: it checks the
incoming batch against the training distribution before trusting the model, and
it records what it decided. A score with no drift check is a score with no idea
whether the model still applies to the population in front of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bundle import ScoringBundle
from .logging_setup import get_logger
from .monitoring.drift import alerts
from .registry import Registry
from .settings import Settings

log = get_logger("scoring")


@dataclass
class ScoringResult:
    scores: pd.DataFrame
    drift: pd.DataFrame
    alerts: list[str]
    run_name: str

    def decision_mix(self) -> pd.Series:
        if "decision" not in self.scores.columns:
            return pd.Series(dtype=float)
        return self.scores["decision"].value_counts(normalize=True).round(4)


def score_batch(
    data: str | Path | pd.DataFrame,
    run: str | Path | None = None,
    registry_root: str | Path = "runs",
    settings: Settings | None = None,
    output: str | Path | None = None,
) -> ScoringResult:
    """Score a batch with a run's bundle, defaulting to whatever is in production."""
    settings = settings or Settings()
    registry = Registry(registry_root)
    resolved = registry.resolve(run)
    bundle = resolved.load_bundle()

    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    log.info(
        "scoring %d row(s) with run '%s' (%s)",
        len(df), resolved.name, bundle.metadata.get("algorithm", "unknown model"),
    )

    drift = bundle.drift_report(df, settings.monitoring)
    messages = alerts(drift, settings.monitoring)
    for message in messages:
        log.warning("drift: %s", message)
    if not messages:
        log.info("drift check passed: the batch matches the training population")

    scores = bundle.score(df)
    mix = scores["decision"].value_counts(normalize=True).round(4).to_dict() if "decision" in scores else {}
    log.info("decisions: %s", mix)

    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(target, index=False)
        log.info("wrote scores to %s", target)
        if len(drift):
            drift_path = target.with_name(f"{target.stem}_drift.csv")
            drift.to_csv(drift_path, index=False)
            log.info("wrote drift report to %s", drift_path)

    return ScoringResult(scores=scores, drift=drift, alerts=messages, run_name=resolved.name)
