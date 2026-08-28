"""Three-sample rule validation.

A rule earns its place only if it holds up on data it was not mined from. Every
candidate is measured on train, test and the sealed holdout, and must clear the
same coverage, lift and hit-count bars on all three — plus a cap on how far its
lift may decay from test to holdout, which is what separates a real risk pocket
from a pattern the mining sample happened to contain.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from ..settings import RuleSettings
from .mining import rule_stats
from .predicates import Rule, evaluate_any

log = get_logger("rules")

DATASETS = ("train", "test", "holdout")


def backtest(
    rules: Sequence[Rule], datasets: Mapping[str, pd.DataFrame], label: str
) -> pd.DataFrame:
    """Long-format performance of every rule on every sample."""
    rows = []
    for name in DATASETS:
        if name not in datasets:
            continue
        for rule in rules:
            rows.append({"dataset": name, **rule_stats(rule, datasets[name], label)})
    return pd.DataFrame(rows)


def stability(results: pd.DataFrame, settings: RuleSettings) -> pd.DataFrame:
    """Pivot the backtest to one row per rule and apply the stability gates."""
    if results.empty:
        return pd.DataFrame()
    wide = results.pivot_table(
        index="rule_id", columns="dataset", values=["coverage", "lift", "hits", "bad_rate"], aggfunc="first"
    )
    wide.columns = [f"{metric}_{dataset}" for metric, dataset in wide.columns]
    wide = wide.reset_index()
    descriptions = results.drop_duplicates("rule_id").set_index("rule_id")[["description", "source"]]
    wide = wide.join(descriptions, on="rule_id")

    verdicts, reasons = [], []
    for _, row in wide.iterrows():
        reason = _first_failure(row, settings)
        verdicts.append(not reason)
        reasons.append(reason or "stable")
    wide["stable"] = verdicts
    wide["verdict"] = reasons
    if {"lift_test", "lift_holdout"}.issubset(wide.columns):
        wide["lift_decay_test_holdout"] = (wide["lift_test"] - wide["lift_holdout"]).round(6)
    sort_key = "lift_holdout" if "lift_holdout" in wide.columns else "lift_train"
    return wide.sort_values(["stable", sort_key], ascending=[False, False]).reset_index(drop=True)


def _first_failure(row: pd.Series, settings: RuleSettings) -> str:
    for dataset in DATASETS:
        hits = row.get(f"hits_{dataset}")
        if hits is None or pd.isna(hits):
            return f"not measured on {dataset}"
        if hits < settings.min_hits:
            return f"only {int(hits)} hit(s) on {dataset} (need {settings.min_hits})"
        coverage = row.get(f"coverage_{dataset}", np.inf)
        if coverage > settings.max_coverage:
            return f"covers {coverage:.2%} of {dataset} (max {settings.max_coverage:.0%})"
        lift = row.get(f"lift_{dataset}", -np.inf)
        if lift < settings.min_lift:
            return f"lift {lift:.2f} on {dataset} (need {settings.min_lift})"
    decay = row.get("lift_test", np.nan) - row.get("lift_holdout", np.nan)
    if pd.notna(decay) and decay > settings.max_lift_decay:
        return f"lift decays {decay:.2f} from test to holdout (max {settings.max_lift_decay})"
    return ""


def select_stable(rules: Sequence[Rule], stability_table: pd.DataFrame) -> list[Rule]:
    if stability_table.empty:
        return []
    passing = set(stability_table.loc[stability_table["stable"], "rule_id"])
    kept = [rule for rule in rules if rule.rule_id in passing]
    log.info("%d of %d rule(s) held up across train, test and holdout", len(kept), len(rules))
    return kept


def combined_effect(
    rules: Sequence[Rule], datasets: Mapping[str, pd.DataFrame], label: str
) -> pd.DataFrame:
    """What the whole rule set does when applied together."""
    rows = []
    for name in DATASETS:
        if name not in datasets:
            continue
        df = datasets[name]
        y = df[label].to_numpy(dtype=float)
        hit = evaluate_any(rules, df) if rules else np.zeros(len(df), dtype=bool)
        rejected = int(hit.sum())
        base = float(y.mean()) if len(y) else 0.0
        rows.append(
            {
                "dataset": name,
                "rules": len(rules),
                "rows": len(df),
                "reject_rate": round(rejected / len(df), 6) if len(df) else 0.0,
                "rejected": rejected,
                "rejected_bad_rate": round(float(y[hit].mean()), 6) if rejected else 0.0,
                "approved_bad_rate": round(float(y[~hit].mean()), 6) if (~hit).any() else np.nan,
                "base_bad_rate": round(base, 6),
                "lift": round(float(y[hit].mean()) / base, 6) if rejected and base else 0.0,
                "bad_rate_reduction": round(base - float(y[~hit].mean()), 6) if (~hit).any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def marginal_contribution(
    rules: Sequence[Rule], df: pd.DataFrame, label: str
) -> pd.DataFrame:
    """What each rule adds on top of the ones before it.

    Rules overlap, so a rule's standalone lift overstates its worth. This applies
    them in order and reports only the rows each one is the first to catch.
    """
    y = df[label].to_numpy(dtype=float)
    base = float(y.mean()) if len(y) else 0.0
    covered = np.zeros(len(df), dtype=bool)
    rows = []
    for rule in rules:
        hit = rule.evaluate(df)
        new = hit & ~covered
        new_hits = int(new.sum())
        covered |= hit
        rows.append(
            {
                "rule_id": rule.rule_id,
                "description": rule.describe(),
                "hits": int(hit.sum()),
                "new_hits": new_hits,
                "new_bad_rate": round(float(y[new].mean()), 6) if new_hits else 0.0,
                "new_lift": round(float(y[new].mean()) / base, 6) if new_hits and base else 0.0,
                "cumulative_reject_rate": round(int(covered.sum()) / len(df), 6) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)
