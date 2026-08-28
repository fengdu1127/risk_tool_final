"""Hard-reject rule discovery.

Two miners feed one pool:

* single-predicate rules — one feature crossing a threshold, or sitting in a
  risky category, or simply being absent;
* tree rules — short conjunctions taken from the leaves of a shallow decision
  tree, which catches risk that only appears where two conditions coincide.

A candidate must be *narrow and sharp*: it may cover at most a few percent of the
book and must concentrate several times the base bad rate. Wide-but-mild
patterns belong in the model, not in a hard cutoff.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..data.schema import DatasetSchema
from ..logging_setup import get_logger
from ..settings import RuleSettings
from .predicates import Predicate, Rule

log = get_logger("rules")


def mine_rules(
    df: pd.DataFrame,
    schema: DatasetSchema,
    features: Sequence[str],
    settings: RuleSettings,
) -> list[Rule]:
    """Mine, filter and de-duplicate candidate rules on the training sample."""
    candidates = mine_single_rules(df, schema, features, settings)
    candidates += mine_tree_rules(df, schema, features, settings)
    scored = [(rule, stats) for rule, stats in ((r, rule_stats(r, df, schema.label)) for r in candidates) if _passes(stats, settings)]
    scored.sort(key=lambda pair: (-pair[1]["lift"], pair[1]["coverage"]))
    kept = _drop_overlapping([rule for rule, _ in scored], df, settings)
    log.info("mined %d candidate rule(s), %d survived filtering and de-duplication", len(candidates), len(kept))
    return kept[: settings.max_rules]


def mine_single_rules(
    df: pd.DataFrame, schema: DatasetSchema, features: Sequence[str], settings: RuleSettings
) -> list[Rule]:
    rules: list[Rule] = []
    for feature in features:
        column = df[feature]
        if column.isna().any():
            rules.append(Rule((Predicate(feature, "is_null"),), source="single"))
        if feature in schema.numeric:
            values = pd.to_numeric(column, errors="coerce").dropna()
            if values.nunique() < 2:
                continue
            for coverage in _coverage_grid(len(values), settings):
                # One threshold per tail, each placed to hit roughly `coverage`.
                upper = float(np.quantile(values, 1.0 - coverage))
                lower = float(np.quantile(values, coverage))
                rules.append(Rule((Predicate(feature, "ge", upper),), source="single"))
                rules.append(Rule((Predicate(feature, "le", lower),), source="single"))
        else:
            for level in column.dropna().astype(str).unique():
                rules.append(Rule((Predicate(feature, "eq", str(level)),), source="single"))
    return rules


def mine_tree_rules(
    df: pd.DataFrame, schema: DatasetSchema, features: Sequence[str], settings: RuleSettings
) -> list[Rule]:
    """Leaf paths of several shallow trees over the numeric features.

    One tree only ever explores the split that looks best first, which buries
    interactions involving its runners-up. Fitting a handful of trees over
    different random feature subsets surfaces those instead — the gates
    downstream decide which of the extra candidates are real.

    Trees are fitted on median-imputed values purely to find candidate
    structure; every candidate is then measured against the real data, where a
    missing value fails the comparison. Any rule that only looked good because of
    imputation therefore fails its own backtest rather than reaching the policy.
    """
    numeric = [f for f in features if f in schema.numeric and f in df.columns]
    if not numeric:
        return []
    try:
        from sklearn.tree import DecisionTreeClassifier, _tree
    except ImportError:
        log.warning("scikit-learn is not installed; skipping tree rule mining")
        return []

    X = df[numeric].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df[schema.label].to_numpy(dtype=float)
    sample = min(settings.tree_feature_sample, len(numeric))

    rules: list[Rule] = []
    seen: set[str] = set()
    for index in range(max(1, settings.tree_count)):
        tree = DecisionTreeClassifier(
            max_depth=settings.tree_max_depth,
            min_samples_leaf=settings.tree_min_leaf_rows,
            # The first tree sees everything; the rest sample, for diversity.
            max_features=None if index == 0 else sample,
            class_weight="balanced",
            random_state=settings.random_state + index,
        ).fit(X, y)

        inner = tree.tree_
        names = [numeric[i] if i != _tree.TREE_UNDEFINED else None for i in inner.feature]

        def walk(node: int, path: list[Predicate]) -> None:
            if inner.feature[node] == _tree.TREE_UNDEFINED:
                if path:
                    rule = Rule(tuple(path), source="tree")
                    if rule.rule_id not in seen:
                        seen.add(rule.rule_id)
                        rules.append(rule)
                return
            feature, threshold = names[node], float(inner.threshold[node])
            walk(inner.children_left[node], path + [Predicate(feature, "le", threshold)])
            walk(inner.children_right[node], path + [Predicate(feature, "gt", threshold)])

        walk(0, [])
    return rules


def _coverage_grid(n_rows: int, settings: RuleSettings) -> np.ndarray:
    """Target coverages to place thresholds at, spread across the usable range.

    The range starts at the smallest slice that can still clear `min_hits` and
    ends at `max_coverage`; spacing is geometric so the narrow, high-lift end
    gets the resolution it needs.
    """
    floor = max(settings.min_hits / max(n_rows, 1), 1e-4)
    ceiling = max(settings.max_coverage, floor * 2)
    points = max(2, settings.grid_points)
    return np.unique(np.geomspace(floor, ceiling, points))


def rule_stats(rule: Rule, df: pd.DataFrame, label: str) -> dict:
    """Coverage, hit count, bad rate and lift of one rule on one sample."""
    y = df[label].to_numpy(dtype=float)
    hit = rule.evaluate(df)
    hits = int(hit.sum())
    base = float(y.mean()) if len(y) else 0.0
    bad_rate = float(y[hit].mean()) if hits else 0.0
    return {
        "rule_id": rule.rule_id,
        "description": rule.describe(),
        "source": rule.source,
        "hits": hits,
        "coverage": round(hits / len(df), 6) if len(df) else 0.0,
        "bad_rate": round(bad_rate, 6),
        "base_bad_rate": round(base, 6),
        "lift": round(bad_rate / base, 6) if base else 0.0,
    }


def _passes(stats: dict, settings: RuleSettings) -> bool:
    return (
        stats["hits"] >= settings.min_hits
        and stats["coverage"] <= settings.max_coverage
        and stats["lift"] >= settings.min_lift
    )


def _drop_overlapping(rules: list[Rule], df: pd.DataFrame, settings: RuleSettings) -> list[Rule]:
    """Keep the strongest of any group of rules that flag mostly the same rows."""
    kept: list[Rule] = []
    masks: list[np.ndarray] = []
    for rule in rules:
        mask = rule.evaluate(df)
        size = int(mask.sum())
        if size == 0:
            continue
        duplicate = False
        for existing in masks:
            smaller = min(size, int(existing.sum()))
            if smaller and int((mask & existing).sum()) / smaller >= settings.max_overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(rule)
            masks.append(mask)
    return kept
