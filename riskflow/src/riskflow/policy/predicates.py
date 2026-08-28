"""Typed rule conditions.

Rules are structured data, not strings. The original temptation is to store a
condition as `"debt_ratio > 0.55"` and re-parse it at scoring time, which breaks
the moment a column name contains a space, a comparison operator, or a non-ASCII
character. Here a condition is a `Predicate` with a named operator, it
serialises to JSON as an object, and the text form exists only for humans.

Missing values never satisfy a comparison. A rule that says "reject when
utilisation is above 0.9" must not fire on an applicant whose utilisation is
unknown — that case belongs to an explicit `is_null` predicate instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..features.binning import as_text, canonical

COMPARISONS = {
    "lt": lambda x, v: x < v,
    "le": lambda x, v: x <= v,
    "gt": lambda x, v: x > v,
    "ge": lambda x, v: x >= v,
}
MEMBERSHIPS = ("in", "not_in")
NULL_CHECKS = ("is_null", "not_null")
OPERATORS = tuple(COMPARISONS) + ("eq", "ne") + MEMBERSHIPS + NULL_CHECKS

_SYMBOLS = {
    "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=",
}


@dataclass(frozen=True)
class Predicate:
    feature: str
    op: str
    value: Any = None

    def __post_init__(self) -> None:
        if self.op not in OPERATORS:
            raise ValueError(f"unknown operator '{self.op}'; expected one of {OPERATORS}")
        if self.op in MEMBERSHIPS and not isinstance(self.value, (list, tuple, set, frozenset)):
            raise ValueError(f"operator '{self.op}' needs a collection of values")
        # Categorical values are normalised once, here, so that both the stored
        # predicate and the column it is later evaluated against are expressed in
        # the same keys. Without this a rule mined on an integer column stops
        # firing the moment pandas reads that column back as float — silently,
        # since a rule that matches nothing raises nothing.
        if self.op in MEMBERSHIPS:
            object.__setattr__(self, "value", tuple(canonical(v) for v in self.value))
        elif self.op in ("eq", "ne"):
            object.__setattr__(self, "value", canonical(self.value))

    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature not in df.columns:
            raise KeyError(f"rule references a column not present in the data: '{self.feature}'")
        column = df[self.feature]
        if self.op == "is_null":
            return column.isna().to_numpy()
        if self.op == "not_null":
            return column.notna().to_numpy()

        present = column.notna().to_numpy()
        if self.op in COMPARISONS:
            numbers = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                hit = COMPARISONS[self.op](numbers, float(self.value))
            return np.where(np.isnan(numbers), False, hit)

        keys = as_text(column)
        if self.op == "eq":
            return np.array([k is not None and k == self.value for k in keys], dtype=bool)
        if self.op == "ne":
            return np.array([k is not None and k != self.value for k in keys], dtype=bool)
        wanted = set(self.value)
        inside = np.array([k is not None and k in wanted for k in keys], dtype=bool)
        return inside if self.op == "in" else (present & ~inside)

    def describe(self) -> str:
        if self.op in NULL_CHECKS:
            return f"{self.feature} {'is missing' if self.op == 'is_null' else 'is present'}"
        if self.op in MEMBERSHIPS:
            values = ", ".join(sorted(str(v) for v in self.value))
            return f"{self.feature} {'in' if self.op == 'in' else 'not in'} ({values})"
        value = self.value
        text = f"{value:.6g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        return f"{self.feature} {_SYMBOLS[self.op]} {text}"

    def to_dict(self) -> dict:
        value = list(self.value) if self.op in MEMBERSHIPS else self.value
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        return {"feature": self.feature, "op": self.op, "value": value}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Predicate":
        value = data.get("value")
        return cls(feature=data["feature"], op=data["op"], value=tuple(value) if data["op"] in MEMBERSHIPS else value)


@dataclass(frozen=True)
class Rule:
    """A conjunction of predicates: every one must hold for the rule to fire."""

    predicates: tuple[Predicate, ...]
    source: str = "single"
    rule_id: str = ""

    def __post_init__(self) -> None:
        if not self.predicates:
            raise ValueError("a rule needs at least one predicate")
        if not self.rule_id:
            object.__setattr__(self, "rule_id", self.describe())

    @property
    def features(self) -> tuple[str, ...]:
        seen: list[str] = []
        for predicate in self.predicates:
            if predicate.feature not in seen:
                seen.append(predicate.feature)
        return tuple(seen)

    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        hit = np.ones(len(df), dtype=bool)
        for predicate in self.predicates:
            hit &= predicate.evaluate(df)
        return hit

    def describe(self) -> str:
        return " AND ".join(p.describe() for p in self.predicates)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "source": self.source,
            "description": self.describe(),
            "predicates": [p.to_dict() for p in self.predicates],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Rule":
        return cls(
            predicates=tuple(Predicate.from_dict(p) for p in data["predicates"]),
            source=data.get("source", "single"),
            rule_id=data.get("rule_id", ""),
        )


def evaluate_any(rules: Sequence[Rule], df: pd.DataFrame) -> np.ndarray:
    """True where at least one rule fires."""
    hit = np.zeros(len(df), dtype=bool)
    for rule in rules:
        hit |= rule.evaluate(df)
    return hit


def first_hit_labels(rules: Sequence[Rule], df: pd.DataFrame) -> np.ndarray:
    """The description of the first rule that fires per row, blank when none do.

    Reporting one reason rather than all of them keeps a declined applicant's
    adverse-action notice readable; the full hit matrix stays available for
    analysis.
    """
    labels = np.full(len(df), "", dtype=object)
    unset = np.ones(len(df), dtype=bool)
    for rule in rules:
        if not unset.any():
            break
        hit = rule.evaluate(df) & unset
        labels[hit] = rule.describe()
        unset &= ~hit
    return labels
