"""Compare two runs.

The question a reviewer actually asks is not "which run has the better AUC" but
"what changed, and would swapping them change anyone's decision". This answers
both: metric deltas, policy differences, and — when a sample is supplied — the
share of applicants whose decision would flip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_setup import get_logger
from .registry import Registry, Run

log = get_logger("compare")

_METRIC_COLUMNS = ("auc", "ks", "gini")


@dataclass
class Comparison:
    metrics: pd.DataFrame
    policy: pd.DataFrame
    decision_shift: pd.DataFrame | None

    def to_text(self) -> str:
        blocks = ["Model metrics", self.metrics.to_string(index=False), "", "Policy", self.policy.to_string(index=False)]
        if self.decision_shift is not None:
            blocks += ["", "Decision shift on the supplied sample", self.decision_shift.to_string(index=False)]
        return "\n".join(blocks)


def compare_runs(
    baseline: str | Path,
    candidate: str | Path,
    registry_root: str | Path = "runs",
    sample: str | Path | pd.DataFrame | None = None,
) -> Comparison:
    registry = Registry(registry_root)
    left, right = registry.get(baseline), registry.get(candidate)

    metrics = _metric_delta(left, right)
    policy = _policy_delta(left, right)
    shift = _decision_shift(left, right, sample) if sample is not None else None
    return Comparison(metrics=metrics, policy=policy, decision_shift=shift)


def _metric_delta(left: Run, right: Run) -> pd.DataFrame:
    def frame(run: Run) -> pd.DataFrame:
        table = run.table("model_metrics")
        if table is None:
            return pd.DataFrame()
        best = run.summary().get("model", {}).get("algorithm")
        return table[table["model"] == best] if best else table

    a, b = frame(left).set_index("dataset"), frame(right).set_index("dataset")
    rows = []
    for dataset in ("train", "test", "holdout"):
        for metric in _METRIC_COLUMNS:
            before = float(a.loc[dataset, metric]) if dataset in a.index and metric in a.columns else np.nan
            after = float(b.loc[dataset, metric]) if dataset in b.index and metric in b.columns else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    left.name: round(before, 6),
                    right.name: round(after, 6),
                    "delta": round(after - before, 6),
                }
            )
    return pd.DataFrame(rows)


def _policy_delta(left: Run, right: Run) -> pd.DataFrame:
    def describe(run: Run) -> dict:
        summary = run.summary()
        policy = summary.get("policy", {})
        cutoff = policy.get("global_cutoff", {})
        holdout = summary.get("outcome_on_holdout", {})
        return {
            "algorithm": summary.get("model", {}).get("algorithm"),
            "reject_at": cutoff.get("reject_at"),
            "review_at": cutoff.get("review_at"),
            "segment_overrides": len(policy.get("segment_overrides", [])),
            "stable_rules": summary.get("rules", {}).get("stable"),
            "holdout_reject_rate": holdout.get("reject_rate"),
            "holdout_approved_bad_rate": holdout.get("approved_bad_rate"),
            "holdout_bad_capture": holdout.get("bad_capture_at_reject"),
        }

    a, b = describe(left), describe(right)
    rows = []
    for key in a:
        before, after = a[key], b[key]
        delta = round(after - before, 6) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else ""
        rows.append({"attribute": key, left.name: before, right.name: after, "delta": delta})
    return pd.DataFrame(rows)


def _decision_shift(left: Run, right: Run, sample) -> pd.DataFrame:
    """Cross-tabulate what each run would decide for the same applicants."""
    df = sample if isinstance(sample, pd.DataFrame) else pd.read_csv(sample)
    before = left.load_bundle().score(df)["decision"]
    after = right.load_bundle().score(df)["decision"]
    table = pd.crosstab(before, after, rownames=[left.name], colnames=[right.name])
    flipped = float((before.to_numpy() != after.to_numpy()).mean())
    log.info("%.2f%% of the sample would be decided differently", flipped * 100)
    out = table.reset_index()
    out.attrs["flipped_share"] = flipped
    return out
