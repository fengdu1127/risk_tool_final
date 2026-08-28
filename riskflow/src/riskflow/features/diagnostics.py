"""Feature diagnostics and screening.

Screening runs as an ordered funnel and records, per feature, the exact gate that
dropped it. Nothing is silently discarded: `screen()` returns a report row for
every candidate feature, whether it survived or not.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..data.schema import DatasetSchema
from ..logging_setup import get_logger
from ..settings import ScreeningSettings
from .binning import MISSING_BIN, Binning, iv_for_bins
from .woe import WoeTransformer, monotonic_correlation

log = get_logger("screening")

_SMOOTH = 1e-6


def psi(reference: np.ndarray | pd.Series, current: np.ndarray | pd.Series, bins: int = 10) -> float:
    """Population stability index over quantile bins of the reference sample."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    return _psi_from_edges(ref, cur, edges)


def _psi_from_edges(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    expected = np.histogram(ref, bins=edges)[0] / len(ref)
    actual = np.histogram(cur, bins=edges)[0] / len(cur)
    expected = np.where(expected == 0, _SMOOTH, expected)
    actual = np.where(actual == 0, _SMOOTH, actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def binned_psi(binning: Binning, reference: pd.Series, current: pd.Series) -> float:
    """PSI computed on the feature's own fitted bins.

    Reusing the modelling bins — rather than re-quantiling each sample — means
    the stability number refers to exactly the buckets the model consumes, and
    it works for categorical features too.
    """
    ref_bins = binning.assign(reference)
    cur_bins = binning.assign(current)
    if ref_bins.size == 0 or cur_bins.size == 0:
        return float("nan")
    levels = np.unique(np.concatenate([ref_bins, cur_bins]))
    expected = np.array([(ref_bins == level).mean() for level in levels], dtype=float)
    actual = np.array([(cur_bins == level).mean() for level in levels], dtype=float)
    expected = np.where(expected == 0, _SMOOTH, expected)
    actual = np.where(actual == 0, _SMOOTH, actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def variance_inflation(woe_df: pd.DataFrame) -> pd.Series:
    """VIF per column, from the inverse correlation matrix.

    Uses the pseudo-inverse so perfectly collinear inputs return large-but-finite
    numbers instead of raising.
    """
    if woe_df.shape[1] < 2:
        return pd.Series(1.0, index=woe_df.columns, dtype=float)
    corr = woe_df.corr().to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    inverse = np.linalg.pinv(corr)
    return pd.Series(np.clip(np.diag(inverse), 1.0, 1e6), index=woe_df.columns)


def correlated_pairs(woe_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """All feature pairs whose absolute correlation reaches `threshold`."""
    if woe_df.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])
    corr = woe_df.corr().abs()
    rows = []
    columns = list(corr.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "abs_corr": round(float(value), 6)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])


def iv_p_value(iv: float, n_rows: int, n_bads: int, n_bins: int) -> float:
    """How often noise alone would produce an information value this large.

    Under the null of no association, `IV * n_bad * n_good / n` follows a
    chi-square law with `n_bins - 1` degrees of freedom — verified by simulation
    across sample sizes, bin counts and base rates. The tail probability is
    evaluated with the Wilson-Hilferty transform, which is accurate well into
    the tail and needs no scipy.

    Returns NaN when the test does not apply, so it never gates on its own.
    """
    degrees = n_bins - 1
    n_goods = n_rows - n_bads
    if degrees < 1 or n_bads <= 0 or n_goods <= 0 or not np.isfinite(iv):
        return float("nan")
    statistic = max(iv, 0.0) * n_bads * n_goods / n_rows
    if statistic <= 0:
        return 1.0
    z = ((statistic / degrees) ** (1 / 3) - (1 - 2 / (9 * degrees))) / np.sqrt(2 / (9 * degrees))
    return float(1.0 - NormalDist().cdf(z))


def benjamini_hochberg(p_values: dict[str, float], alpha: float) -> set[str]:
    """Names whose evidence survives false-discovery-rate control.

    Bonferroni is the wrong tool for feature screening: with three hundred
    candidates it demands significance no realistic sample can supply, and with
    a dozen it is needlessly strict. Controlling the false discovery rate adapts
    instead — lenient when few features are tested, sharp when most of a large
    candidate set is noise, which is exactly the regime a feature store creates.
    """
    testable = {k: v for k, v in p_values.items() if v is not None and np.isfinite(v)}
    if alpha <= 0 or not testable:
        return set(p_values)
    ordered = sorted(testable.items(), key=lambda kv: kv[1])
    m = len(ordered)
    cutoff_rank = 0
    for rank, (_, p) in enumerate(ordered, start=1):
        if p <= alpha * rank / m:
            cutoff_rank = rank
    survivors = {name for name, _ in ordered[:cutoff_rank]}
    # Features the test could not be applied to are not condemned by it.
    survivors |= {k for k in p_values if k not in testable}
    return survivors


@dataclass(frozen=True)
class ScreeningResult:
    selected: tuple[str, ...]
    report: pd.DataFrame
    correlated: pd.DataFrame

    def dropped(self) -> pd.DataFrame:
        return self.report[~self.report["selected"]].reset_index(drop=True)


def screen(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    schema: DatasetSchema,
    woe: WoeTransformer,
    settings: ScreeningSettings,
) -> ScreeningResult:
    """Run the screening funnel and return the surviving features plus the audit trail."""
    rows: dict[str, dict] = {}
    for feature, reason in woe.skipped.items():
        rows[feature] = {"feature": feature, "kind": "-", "selected": False, "reason": reason}

    holdout_rows = len(test) if test is not None else 0
    holdout_bads = int(test[schema.label].sum()) if holdout_rows else 0

    survivors: list[str] = []
    for feature in woe.features:
        binning = woe.binnings[feature]
        missing_rate = float(train[feature].isna().mean()) if feature in train.columns else 0.0
        mono = monotonic_correlation(binning)
        stability = (
            binned_psi(binning, train[feature], test[feature])
            if test is not None and len(test) and feature in test.columns
            else float("nan")
        )
        # IV measured where the bins were chosen is optimistically biased: with
        # enough candidates, supervised binning will carve apparent signal out of
        # pure noise. Re-measuring on held-out data with the *same* bins is what
        # separates a real driver from a lucky split.
        held_out_iv = (
            iv_for_bins(binning.assign(test[feature]), test[schema.label])
            if test is not None and len(test) and feature in test.columns
            else float("nan")
        )
        p_value = (
            iv_p_value(held_out_iv, holdout_rows, holdout_bads, len(binning.stats))
            if holdout_rows
            else float("nan")
        )
        row = {
            "feature": feature,
            "kind": binning.kind,
            "missing_rate": round(missing_rate, 6),
            "iv": binning.iv,
            "iv_holdout": None if np.isnan(held_out_iv) else round(held_out_iv, 6),
            "iv_p_value": None if np.isnan(p_value) else float(f"{p_value:.3g}"),
            "n_bins": len(binning.stats),
            "monotonic_corr": None if np.isnan(mono) else round(mono, 6),
            "psi_train_test": None if np.isnan(stability) else round(stability, 6),
            "selected": True,
            "reason": "",
        }

        reason = _first_failure(row, settings, binning.kind)
        if reason:
            row["selected"] = False
            row["reason"] = reason
        else:
            survivors.append(feature)
        rows[feature] = row

    # Significance is a property of the candidate set, not of one feature, so it
    # is decided once across everything that got this far. Off by default: see
    # `iv_significance_alpha` for why.
    if survivors and holdout_rows and settings.iv_significance_alpha > 0:
        significant = benjamini_hochberg(
            {f: rows[f].get("iv_p_value") for f in survivors}, settings.iv_significance_alpha
        )
        for feature in list(survivors):
            if feature not in significant:
                survivors.remove(feature)
                rows[feature].update(
                    selected=False,
                    reason=(
                        f"out-of-sample IV is within what noise produces here "
                        f"(p={rows[feature]['iv_p_value']}, FDR alpha={settings.iv_significance_alpha})"
                    ),
                )

    # Redundancy gates need the surviving set, so they run after the per-feature funnel.
    woe_train = woe.transform(train, tuple(survivors)) if survivors else pd.DataFrame()
    correlated = correlated_pairs(woe_train, settings.max_abs_corr) if survivors else pd.DataFrame()
    iv_by_feature = {f: woe.binnings[f].iv for f in survivors}

    for _, pair in correlated.iterrows():
        a, b = pair["feature_a"], pair["feature_b"]
        if a not in survivors or b not in survivors:
            continue
        weaker, stronger = (b, a) if iv_by_feature[a] >= iv_by_feature[b] else (a, b)
        survivors.remove(weaker)
        rows[weaker].update(
            selected=False,
            reason=f"correlated with {stronger} (|r|={pair['abs_corr']:.3f}, lower IV)",
        )

    while len(survivors) >= 2:
        vif = variance_inflation(woe.transform(train, tuple(survivors)))
        worst = vif.idxmax()
        if vif[worst] <= settings.max_vif:
            break
        survivors.remove(worst)
        rows[worst].update(selected=False, reason=f"VIF {vif[worst]:.1f} above {settings.max_vif}")

    for feature in survivors:
        rows[feature]["selected"] = True
        rows[feature]["reason"] = "selected"

    report = pd.DataFrame(list(rows.values()))
    if len(report):
        report = report.sort_values(["selected", "iv"], ascending=[False, False]).reset_index(drop=True)
    log.info("screening kept %d of %d feature(s)", len(survivors), len(rows))
    if not survivors:
        raise ValueError(
            "screening rejected every feature; loosen ScreeningSettings "
            "(min_iv / min_monotonic_corr / max_psi) or check the label"
        )
    return ScreeningResult(selected=tuple(survivors), report=report, correlated=correlated)


def _first_failure(row: dict, settings: ScreeningSettings, kind: str) -> str:
    if row["missing_rate"] > settings.max_missing_rate:
        return f"missing rate {row['missing_rate']:.1%} above {settings.max_missing_rate:.0%}"
    if row["iv"] < settings.min_iv:
        return f"IV {row['iv']:.4f} below {settings.min_iv}"
    if row["iv"] > settings.max_iv:
        return f"IV {row['iv']:.4f} above {settings.max_iv} — check for label leakage"
    mono = row["monotonic_corr"]
    if kind == "numeric" and mono is not None and abs(mono) < settings.min_monotonic_corr:
        return f"non-monotone risk (corr={mono:.3f}, need |corr|>={settings.min_monotonic_corr})"
    stability = row["psi_train_test"]
    if stability is not None and stability > settings.max_psi:
        return f"unstable between train and test (PSI={stability:.3f} > {settings.max_psi})"
    return ""
