"""The decision policy: what actually happens to an application.

This is the object the scoring service consumes. It is deliberately small and
fully declarative — rules, a global cutoff, and any per-segment overrides — so a
policy can be read, diffed and reviewed as JSON without running anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .predicates import Rule, first_hit_labels
from .thresholds import Cutoff, SegmentCutoff

APPROVE, REVIEW, REJECT = "approve", "review", "reject"

# Scores are compared to thresholds at this many decimal places.
#
# A dot product is not bit-reproducible across batch sizes: BLAS picks a
# different summation order for a one-row matrix than for a thousand-row one, so
# the same applicant's score can differ in its last bit between a real-time call
# and a batch reconciliation. Thresholds are empirical quantiles of training
# scores — literally equal to observed values — so a row can sit exactly on one,
# and measurement showed 26 decisions per 300 such thresholds flipping between
# the two paths. Comparing at a fixed precision removes that: 1e-12 is far below
# any meaningful resolution of a probability and far above the ~1e-16 noise.
DECISION_PRECISION = 12


def _at_decision_precision(values: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float), DECISION_PRECISION)


@dataclass(frozen=True)
class DecisionPolicy:
    global_cutoff: Cutoff
    reject_rules: tuple[Rule, ...] = ()
    segment_cutoffs: tuple[SegmentCutoff, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def thresholds_for(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row reject / review thresholds and the segment each row matched.

        Later overrides win, so a row matching two segments takes the last one
        listed; the policy is built with at most one override per segment value,
        which keeps that from mattering in practice.
        """
        reject_at = np.full(len(df), self.global_cutoff.reject_at, dtype=float)
        review_at = np.full(len(df), self.global_cutoff.review_at, dtype=float)
        segment = np.full(len(df), "", dtype=object)
        for override in self.segment_cutoffs:
            mask = override.matches(df)
            reject_at[mask] = override.cutoff.reject_at
            review_at[mask] = override.cutoff.review_at
            segment[mask] = f"{override.feature}={override.value if override.value is not None else '__missing__'}"
        return reject_at, review_at, segment

    def required_columns(self) -> tuple[str, ...]:
        """Columns the policy itself needs, beyond the model's own inputs.

        A rule or a segment override can reference a column the model never
        used. If one of those is absent at scoring time the policy quietly
        stops applying to those rows, so callers must be able to check for
        them up front rather than discover it from a shifted decision mix.
        """
        needed: list[str] = []
        for rule in self.reject_rules:
            needed.extend(rule.features)
        needed.extend(override.feature for override in self.segment_cutoffs)
        seen: list[str] = []
        for column in needed:
            if column not in seen:
                seen.append(column)
        return tuple(seen)

    def decide(self, df: pd.DataFrame, scores) -> pd.DataFrame:
        """Apply the policy. A hard rule outranks the score bands."""
        score = np.asarray(pd.Series(scores).to_numpy(), dtype=float)
        if len(score) != len(df):
            raise ValueError(f"got {len(score)} scores for {len(df)} rows")
        # A non-finite score compares false against every threshold, which would
        # quietly approve the applicant. Failing open is the one outcome a
        # credit decision must never have, so this is an error, not a default.
        unusable = ~np.isfinite(score)
        if unusable.any():
            raise ValueError(
                f"{int(unusable.sum())} of {len(score)} scores are not finite "
                f"(first at row {int(np.argmax(unusable))}); refusing to decide on them"
            )

        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            raise KeyError(
                f"the policy references column(s) absent from the data: {missing}; "
                "without them its rules and segment overrides would silently stop applying"
            )

        reject_at, review_at, segment = self.thresholds_for(df)
        score = _at_decision_precision(score)
        reject_at = _at_decision_precision(reject_at)
        review_at = _at_decision_precision(review_at)
        rule_hit = first_hit_labels(self.reject_rules, df) if self.reject_rules else np.full(len(df), "", dtype=object)
        by_rule = rule_hit != ""
        by_score = score >= reject_at
        in_review = (score >= review_at) & ~by_score

        decision = np.full(len(df), APPROVE, dtype=object)
        decision[in_review] = REVIEW
        decision[by_score | by_rule] = REJECT

        reason = np.full(len(df), "score below the review threshold", dtype=object)
        reason[in_review] = "score in the review band"
        reason[by_score] = "score at or above the reject threshold"
        reason[by_rule] = np.array(["rule: " + label for label in rule_hit[by_rule]], dtype=object)

        return pd.DataFrame(
            {
                "decision": decision,
                "reason": reason,
                "rejected_by_rule": by_rule,
                "rejected_by_score": by_score,
                "rule_hit": rule_hit,
                "segment": segment,
                "reject_at": reject_at,
                "review_at": review_at,
            },
            index=df.index,
        )

    def summarise(self, df: pd.DataFrame, scores, label: str | None = None) -> pd.DataFrame:
        """Decision mix, and the resulting bad rates when labels are available."""
        decisions = self.decide(df, scores)["decision"]
        rows = []
        y = df[label].to_numpy(dtype=float) if label and label in df.columns else None
        base = float(y.mean()) if y is not None and len(y) else None
        for action in (REJECT, REVIEW, APPROVE):
            mask = (decisions == action).to_numpy()
            row = {"action": action, "rows": int(mask.sum()), "share": round(float(mask.mean()), 6)}
            if y is not None:
                bad_rate = float(y[mask].mean()) if mask.any() else 0.0
                row["bad_rate"] = round(bad_rate, 6)
                row["lift"] = round(bad_rate / base, 6) if base else 0.0
                row["bad_capture"] = round(float(y[mask].sum()) / y.sum(), 6) if y.sum() else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "global_cutoff": self.global_cutoff.to_dict(),
            "segment_cutoffs": [s.to_dict() for s in self.segment_cutoffs],
            "reject_rules": [r.to_dict() for r in self.reject_rules],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "DecisionPolicy":
        return cls(
            global_cutoff=Cutoff.from_dict(data["global_cutoff"]),
            reject_rules=tuple(Rule.from_dict(r) for r in data.get("reject_rules", [])),
            segment_cutoffs=tuple(SegmentCutoff.from_dict(s) for s in data.get("segment_cutoffs", [])),
            created_at=data.get("created_at", ""),
        )
