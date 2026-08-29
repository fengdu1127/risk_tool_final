"""Score cutoff search, globally and per segment.

One principle governs which sample is allowed to do what:

    **the holdout may veto, never select.**

Cutoffs are chosen on test. The holdout is then consulted only to strike down a
choice that fails to reproduce — it can shrink the policy, never shape it. That
keeps the final validation honest while still refusing to ship a threshold that
only worked once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..features.binning import as_text
from ..logging_setup import get_logger
from ..settings import CutoffSettings

log = get_logger("cutoffs")

DATASETS = ("train", "test", "holdout")


@dataclass(frozen=True)
class Cutoff:
    """Score bands. Higher score means higher risk."""

    reject_at: float
    review_at: float

    def decide(self, scores: np.ndarray) -> np.ndarray:
        # Same fixed precision the deployed policy uses, so a cutoff evaluated
        # during training and applied in production agree row for row.
        from .decision import DECISION_PRECISION

        scores = np.round(np.asarray(scores, dtype=float), DECISION_PRECISION)
        decision = np.full(len(scores), "approve", dtype=object)
        decision[scores >= round(self.review_at, DECISION_PRECISION)] = "review"
        decision[scores >= round(self.reject_at, DECISION_PRECISION)] = "reject"
        return decision

    def to_dict(self) -> dict:
        return {"reject_at": float(self.reject_at), "review_at": float(self.review_at)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Cutoff":
        return cls(reject_at=float(data["reject_at"]), review_at=float(data["review_at"]))


@dataclass(frozen=True)
class SegmentCutoff:
    """A cutoff that replaces the global one for rows in a given segment."""

    feature: str
    value: str | None  # None matches missing values
    cutoff: Cutoff

    def matches(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature not in df.columns:
            return np.zeros(len(df), dtype=bool)
        # Canonical keys, so a segment identified as `term=12` during training
        # still matches when the column arrives as float 12.0 at scoring time.
        keys = as_text(df[self.feature])
        return np.array([key == self.value for key in keys], dtype=bool)

    def to_dict(self) -> dict:
        return {"feature": self.feature, "value": self.value, "cutoff": self.cutoff.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping) -> "SegmentCutoff":
        return cls(feature=data["feature"], value=data["value"], cutoff=Cutoff.from_dict(data["cutoff"]))


@dataclass
class CutoffSearch:
    chosen: Cutoff
    candidates: pd.DataFrame
    performance: pd.DataFrame


def evaluate_cutoff(
    cutoff: Cutoff,
    datasets: Mapping[str, pd.DataFrame],
    scores: Mapping[str, np.ndarray],
    label: str,
    tag: str = "",
) -> pd.DataFrame:
    """Volume, bad rate and lift of each decision band, on each sample."""
    rows = []
    for name in DATASETS:
        if name not in datasets:
            continue
        df, score = datasets[name], np.asarray(scores[name], dtype=float)
        y = df[label].to_numpy(dtype=float)
        if len(y) == 0:
            continue
        base = float(y.mean())
        decision = cutoff.decide(score)
        for action in ("reject", "review", "approve"):
            mask = decision == action
            count = int(mask.sum())
            bad_rate = float(y[mask].mean()) if count else 0.0
            rows.append(
                {
                    "policy": tag,
                    "dataset": name,
                    "action": action,
                    "reject_at": cutoff.reject_at,
                    "review_at": cutoff.review_at,
                    "rows": count,
                    "share": round(count / len(y), 6),
                    "bad_rate": round(bad_rate, 6),
                    "base_bad_rate": round(base, 6),
                    "lift": round(bad_rate / base, 6) if base else 0.0,
                    "bad_capture": round(float(y[mask].sum()) / y.sum(), 6) if y.sum() else 0.0,
                }
            )
    return pd.DataFrame(rows)


def search_global_cutoff(
    datasets: Mapping[str, pd.DataFrame],
    scores: Mapping[str, np.ndarray],
    label: str,
    settings: CutoffSettings,
) -> CutoffSearch:
    """Pick the widest reject and review bands that still concentrate risk.

    Thresholds come from the training score distribution so they are reproducible
    at scoring time; which one to use is decided on test.
    """
    train_scores = np.asarray(scores["train"], dtype=float)
    candidate_frames, options = [], []
    for reject_rate in sorted(settings.reject_rate_grid):
        reject_at = float(np.quantile(train_scores, 1.0 - reject_rate))
        for review_rate in sorted(settings.review_rate_grid):
            review_at = float(np.quantile(train_scores, max(0.0, 1.0 - reject_rate - review_rate)))
            if review_at > reject_at:
                continue
            cutoff = Cutoff(reject_at=reject_at, review_at=review_at)
            tag = f"reject{reject_rate:.0%}_review{review_rate:.0%}".replace("%", "pct")
            frame = evaluate_cutoff(cutoff, datasets, scores, label, tag)
            frame["target_reject_rate"] = reject_rate
            frame["target_review_rate"] = review_rate
            candidate_frames.append(frame)
            options.append((reject_rate, review_rate, cutoff, tag))

    if not options:
        raise ValueError("no valid cutoff candidates; check reject/review rate grids")
    candidates = pd.concat(candidate_frames, ignore_index=True)

    reject_rate = _widest_qualifying(candidates, "reject", settings.min_reject_lift, "target_reject_rate")
    within = candidates[candidates["target_reject_rate"] == reject_rate]
    review_rate = _widest_qualifying(within, "review", settings.min_review_lift, "target_review_rate")
    chosen = next(c for r, v, c, _ in options if r == reject_rate and v == review_rate)
    log.info(
        "chose reject rate %.0f%% / review rate %.0f%% on test (score >= %.4f rejects)",
        reject_rate * 100, review_rate * 100, chosen.reject_at,
    )
    performance = evaluate_cutoff(chosen, datasets, scores, label, "chosen")
    return CutoffSearch(chosen=chosen, candidates=candidates, performance=performance)


def _widest_qualifying(candidates: pd.DataFrame, action: str, min_lift: float, rate_column: str) -> float:
    """Largest band whose lift on test still clears `min_lift`."""
    on_test = candidates[(candidates["dataset"] == "test") & (candidates["action"] == action)]
    if on_test.empty:
        raise ValueError(f"no test-set results for the '{action}' band")
    qualifying = on_test[on_test["lift"] >= min_lift]
    if qualifying.empty:
        # Nothing clears the bar: fall back to the narrowest band, which is the
        # most selective one available, and say so.
        fallback = float(on_test[rate_column].min())
        log.warning(
            "no %s band reaches lift %.2f on test; falling back to the narrowest (%.0f%%)",
            action, min_lift, fallback * 100,
        )
        return fallback
    return float(qualifying[rate_column].max())


def search_segment_cutoffs(
    datasets: Mapping[str, pd.DataFrame],
    scores: Mapping[str, np.ndarray],
    label: str,
    global_cutoff: Cutoff,
    settings: CutoffSettings,
) -> tuple[list[SegmentCutoff], pd.DataFrame]:
    """Per-segment reject thresholds, chosen on test and vetoed by holdout.

    A segment only gets its own threshold when it is large enough to measure,
    genuinely differs in risk from the book, and reproduces on the holdout. The
    review threshold stays global so applicants face one consistent soft band.
    """
    rows: list[dict] = []
    overrides: list[SegmentCutoff] = []
    train, test = datasets["train"], datasets.get("test")
    holdout = datasets.get("holdout")

    for feature in settings.segment_features:
        if feature not in train.columns:
            log.warning("segment feature '%s' is not in the data; skipping", feature)
            continue
        base_bad_rate = float(train[label].mean())
        for value in _segment_values(train[feature]):
            segment = SegmentCutoff(feature=feature, value=value, cutoff=global_cutoff)
            train_mask = segment.matches(train)
            record = {
                "feature": feature,
                "value": "__missing__" if value is None else value,
                "train_rows": int(train_mask.sum()),
                "train_share": round(float(train_mask.mean()), 6),
                "train_bad_rate": round(float(train[label].to_numpy(float)[train_mask].mean()), 6) if train_mask.any() else np.nan,
            }
            record["bad_rate_gap"] = round(abs(record["train_bad_rate"] - base_bad_rate), 6) if train_mask.any() else np.nan

            veto = _segment_veto(record, settings)
            if veto:
                rows.append({**record, "accepted": False, "verdict": veto})
                continue

            best = _best_segment_cutoff(
                feature, value, train, test, scores, label, global_cutoff, settings
            )
            if best is None:
                rows.append({**record, "accepted": False, "verdict": "no candidate threshold cleared the lift bar on test"})
                continue
            candidate, test_lift, target_rate = best
            record.update({"reject_at": candidate.cutoff.reject_at, "target_reject_rate": target_rate, "test_lift": round(test_lift, 6)})

            verdict = _holdout_veto(candidate, holdout, scores.get("holdout"), label, test_lift, settings)
            record["accepted"] = not verdict
            record["verdict"] = verdict or "accepted"
            if not verdict:
                overrides.append(candidate)
            rows.append(record)

    if overrides:
        log.info("accepted %d segment override(s)", len(overrides))
    return overrides, pd.DataFrame(rows)


def _segment_values(column: pd.Series) -> list[str | None]:
    """Distinct segment keys, using the same canonical form matching uses."""
    keys = as_text(column)
    values: list[str | None] = sorted({k for k in keys if k is not None})
    if any(k is None for k in keys):
        values.append(None)
    return values


def _segment_veto(record: dict, settings: CutoffSettings) -> str:
    if record["train_rows"] < settings.segment_min_rows:
        return f"only {record['train_rows']} training rows (need {settings.segment_min_rows})"
    if record["train_share"] < settings.segment_min_share:
        return f"segment is {record['train_share']:.1%} of the book (need {settings.segment_min_share:.0%})"
    if pd.isna(record["bad_rate_gap"]) or record["bad_rate_gap"] < settings.segment_min_bad_rate_gap:
        return f"risk differs from the book by only {record['bad_rate_gap']:.3f} (need {settings.segment_min_bad_rate_gap})"
    return ""


def _best_segment_cutoff(
    feature: str,
    value: str | None,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    scores: Mapping[str, np.ndarray],
    label: str,
    global_cutoff: Cutoff,
    settings: CutoffSettings,
) -> tuple[SegmentCutoff, float, float] | None:
    probe = SegmentCutoff(feature=feature, value=value, cutoff=global_cutoff)
    train_mask = probe.matches(train)
    train_scores = np.asarray(scores["train"], dtype=float)[train_mask]
    if test is None or len(train_scores) == 0:
        return None
    test_mask = probe.matches(test)
    test_scores = np.asarray(scores["test"], dtype=float)[test_mask]
    y_test = test[label].to_numpy(dtype=float)[test_mask]
    if len(y_test) == 0 or y_test.mean() == 0:
        return None

    best = None
    for rate in sorted(settings.reject_rate_grid):
        reject_at = float(np.quantile(train_scores, 1.0 - rate))
        hit = test_scores >= reject_at
        if hit.sum() == 0:
            continue
        lift = float(y_test[hit].mean() / y_test.mean())
        if lift < settings.segment_min_lift:
            continue
        # Widest qualifying band wins, matching the global search.
        candidate = SegmentCutoff(
            feature=feature,
            value=value,
            cutoff=Cutoff(reject_at=reject_at, review_at=min(global_cutoff.review_at, reject_at)),
        )
        best = (candidate, lift, rate)
    return best


def _holdout_veto(
    segment: SegmentCutoff,
    holdout: pd.DataFrame | None,
    holdout_scores: np.ndarray | None,
    label: str,
    test_lift: float,
    settings: CutoffSettings,
) -> str:
    if holdout is None or holdout_scores is None:
        return ""
    mask = segment.matches(holdout)
    y = holdout[label].to_numpy(dtype=float)[mask]
    scores = np.asarray(holdout_scores, dtype=float)[mask]
    if len(y) == 0 or y.mean() == 0:
        return "segment is absent from the holdout"
    hit = scores >= segment.cutoff.reject_at
    if int(hit.sum()) < settings.segment_min_holdout_hits:
        return f"only {int(hit.sum())} holdout rejects (need {settings.segment_min_holdout_hits})"
    lift = float(y[hit].mean() / y.mean())
    if abs(test_lift - lift) > settings.segment_max_lift_gap:
        return f"lift moves {test_lift:.2f} to {lift:.2f} between test and holdout"
    return ""
