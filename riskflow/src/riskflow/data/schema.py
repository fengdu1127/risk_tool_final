"""Dataset schema: which column is the label, which columns are usable features.

The schema is resolved once, persisted into the scoring bundle, and re-checked at
scoring time. That is what stops a column silently changing dtype between
training and production.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import pandas as pd

from ..logging_setup import get_logger

log = get_logger("schema")

# Columns that are structurally unusable as model features.
_RESERVED_SUFFIXES = ("_id",)


@dataclass(frozen=True)
class DatasetSchema:
    label: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    time_col: str | None = None
    id_col: str | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def kind(self, feature: str) -> str:
        if feature in self.numeric:
            return "numeric"
        if feature in self.categorical:
            return "categorical"
        raise KeyError(f"'{feature}' is not a feature in this schema")

    def subset(self, features: Sequence[str]) -> "DatasetSchema":
        keep = set(features)
        return DatasetSchema(
            label=self.label,
            numeric=tuple(c for c in self.numeric if c in keep),
            categorical=tuple(c for c in self.categorical if c in keep),
            time_col=self.time_col,
            id_col=self.id_col,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "DatasetSchema":
        return cls(
            label=data["label"],
            numeric=tuple(data.get("numeric", ())),
            categorical=tuple(data.get("categorical", ())),
            time_col=data.get("time_col"),
            id_col=data.get("id_col"),
        )


def infer_schema(
    df: pd.DataFrame,
    label: str,
    features: Sequence[str] | None = None,
    time_col: str | None = None,
    id_col: str | None = None,
) -> DatasetSchema:
    """Split the frame's columns into numeric and categorical features.

    Booleans are treated as numeric; anything else non-numeric is categorical.
    The label, time and id columns are never features.
    """
    _reject_duplicate_columns(df)
    if label not in df.columns:
        raise ValueError(f"label column '{label}' not in data; columns are {list(df.columns)}")
    for name, col in (("time_col", time_col), ("id_col", id_col)):
        if col is not None and col not in df.columns:
            raise ValueError(f"{name} '{col}' not in data")

    excluded = {label, time_col, id_col} - {None}
    if features is None:
        candidates = [c for c in df.columns if c not in excluded]
        candidates = [c for c in candidates if not c.lower().endswith(_RESERVED_SUFFIXES)]
    else:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"requested features not in data: {missing}")
        overlap = [c for c in features if c in excluded]
        if overlap:
            raise ValueError(f"these columns cannot be features: {overlap}")
        candidates = list(features)

    numeric, categorical = [], []
    for col in candidates:
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            log.warning("dropping datetime column '%s'; pass it as time_col to use it", col)
        else:
            categorical.append(col)

    if not numeric and not categorical:
        raise ValueError("no usable feature columns were found")
    return DatasetSchema(
        label=label,
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        time_col=time_col,
        id_col=id_col,
    )


def _reject_duplicate_columns(df: pd.DataFrame) -> None:
    """Duplicate column names make `df[col]` return a frame, not a series.

    Joins produce these routinely, and every downstream operation then fails
    somewhere deep with an unrelated-looking pandas error about the truth value
    of a Series.
    """
    duplicated = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
    if duplicated:
        raise ValueError(f"duplicate column name(s) in the data: {duplicated}")


def coerce_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a frame whose label column is numeric 0/1.

    CSV round trips routinely turn a label into the strings "0" and "1", and
    booleans are just as common. Both are unambiguous, so they are converted
    rather than rejected; anything else is refused by name.
    """
    if label not in df.columns:
        raise ValueError(f"label column '{label}' not in data")
    column = df[label]
    if pd.api.types.is_bool_dtype(column):
        return df.assign(**{label: column.astype(float)})
    if pd.api.types.is_numeric_dtype(column):
        return df
    converted = pd.to_numeric(column, errors="coerce")
    unconvertible = converted.isna() & column.notna()
    if unconvertible.any():
        offenders = sorted({str(v) for v in column[unconvertible].unique()})[:5]
        raise ValueError(
            f"label '{label}' holds non-numeric value(s) {offenders}; it must be 0/1"
        )
    return df.assign(**{label: converted.astype(float)})


def validate_frame(df: pd.DataFrame, schema: DatasetSchema, *, require_label: bool = True) -> list[str]:
    """Check a frame against the schema.

    Raises on problems that make the run meaningless; returns warnings for the
    rest so the caller can log them.
    """
    warnings: list[str] = []
    _reject_duplicate_columns(df)
    missing = [c for c in schema.features if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing schema features: {missing}")

    if require_label:
        if schema.label not in df.columns:
            raise ValueError(f"label column '{schema.label}' not in data")
        labels = df[schema.label].dropna().unique()
        if len(labels) < 2:
            raise ValueError(f"label '{schema.label}' has a single class: {list(labels)}")
        unexpected = set(map(float, labels)) - {0.0, 1.0}
        if unexpected:
            raise ValueError(f"label '{schema.label}' must be 0/1, found {sorted(unexpected)}")
        bad_rate = float(df[schema.label].mean())
        if bad_rate < 0.005 or bad_rate > 0.995:
            warnings.append(f"extreme class balance: bad rate {bad_rate:.3%}")
        if df[schema.label].isna().any():
            warnings.append(f"{int(df[schema.label].isna().sum())} rows have a null label and will be dropped")

    if len(df) < 200:
        warnings.append(f"only {len(df)} rows; estimates will be noisy")

    # A handful of rows is constant by definition, so only flag it once the
    # sample is big enough for constancy to mean something.
    if len(df) >= 30:
        for col in schema.features:
            if df[col].nunique(dropna=True) <= 1:
                warnings.append(f"feature '{col}' is constant")
    for col in schema.numeric:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            if not pd.api.types.is_bool_dtype(df[col]):
                warnings.append(f"feature '{col}' was numeric at training time but is now {df[col].dtype}")
    return warnings
