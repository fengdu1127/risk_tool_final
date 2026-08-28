"""Run figures. Every plot answers a question a reviewer will actually ask."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger("report")

_PALETTE = {"train": "#4c78a8", "test": "#f58518", "holdout": "#54a24b"}


def _pyplot(dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": dpi, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    return plt


def make_figures(run, dpi: int = 120) -> list[Path]:
    """Render every figure the run has data for; skip the rest quietly."""
    try:
        plt = _pyplot(dpi)
    except ImportError:
        log.warning("matplotlib is not installed; skipping figures")
        return []

    run.figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in (
        ("information_value", _iv_plot),
        ("gains_by_band", _gains_plot),
        ("calibration", _calibration_plot),
        ("cutoff_tradeoff", _cutoff_plot),
        ("woe_trends", _woe_plot),
    ):
        try:
            figure = builder(run, plt)
        except Exception as exc:
            log.debug("figure '%s' skipped: %s", name, exc)
            continue
        if figure is None:
            continue
        path = run.figures / f"{name}.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    log.info("wrote %d figure(s)", len(written))
    return written


def _iv_plot(run, plt):
    table = run.table("iv")
    if table is None or table.empty:
        return None
    top = table.head(15).iloc[::-1]
    figure, ax = plt.subplots(figsize=(6.5, max(2.5, 0.32 * len(top))))
    ax.barh(top["feature"], top["iv"], color="#4c78a8")
    ax.set_xlabel("information value")
    ax.set_title("Predictive strength by feature")
    return figure


def _gains_plot(run, plt):
    table = run.table("gains")
    if table is None or table.empty:
        return None
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for dataset, group in table.groupby("dataset"):
        group = group.sort_values("band")
        colour = _PALETTE.get(dataset, "#888888")
        left.plot(group["band"], group["bad_rate"], marker="o", label=dataset, color=colour)
        right.plot(group["cum_share"], group["cum_bad_capture"], marker="o", label=dataset, color=colour)
    left.set_xlabel("score band (higher = riskier)")
    left.set_ylabel("bad rate")
    left.set_title("Bad rate by score band")
    left.legend()
    right.plot([0, 1], [0, 1], "--", color="#bbbbbb", linewidth=1)
    right.set_xlabel("share of applicants declined")
    right.set_ylabel("share of bads caught")
    right.set_title("Bad capture")
    right.legend()
    return figure


def _calibration_plot(run, plt):
    table = run.table("calibration")
    if table is None or table.empty:
        return None
    figure, ax = plt.subplots(figsize=(4.6, 4.2))
    limit = 0.0
    for dataset, group in table.groupby("dataset"):
        ax.scatter(group["calibrated_mean"], group["actual_bad_rate"], label=dataset, s=28, color=_PALETTE.get(dataset, "#888888"))
        limit = max(limit, float(group[["calibrated_mean", "actual_bad_rate"]].to_numpy().max()))
    limit = limit * 1.1 or 1.0
    ax.plot([0, limit], [0, limit], "--", color="#bbbbbb", linewidth=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("predicted bad rate (calibrated)")
    ax.set_ylabel("observed bad rate")
    ax.set_title("Calibration")
    ax.legend()
    return figure


def _cutoff_plot(run, plt):
    table = run.table("cutoff_candidates")
    if table is None or table.empty:
        return None
    rejected = table[(table["action"] == "reject") & (table["dataset"] == "test")]
    approved = table[(table["action"] == "approve") & (table["dataset"] == "test")]
    if rejected.empty or approved.empty:
        return None
    merged = rejected.merge(approved, on="policy", suffixes=("_reject", "_approve"))
    chosen = run.summary().get("policy", {}).get("global_cutoff", {})

    figure, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.scatter(merged["share_reject"], merged["bad_rate_approve"], s=26, color="#4c78a8", label="candidate cutoffs")
    if chosen:
        performance = run.table("cutoff_performance")
        if performance is not None and not performance.empty:
            on_test = performance[performance["dataset"] == "test"]
            pick_reject = on_test[on_test["action"] == "reject"]
            pick_approve = on_test[on_test["action"] == "approve"]
            if len(pick_reject) and len(pick_approve):
                ax.scatter(
                    pick_reject["share"].iloc[0], pick_approve["bad_rate"].iloc[0],
                    s=110, marker="*", color="#e45756", zorder=5, label="chosen",
                )
    ax.set_xlabel("share declined")
    ax.set_ylabel("bad rate among approvals")
    ax.set_title("Cutoff trade-off (test)")
    ax.legend()
    return figure


def _woe_plot(run, plt):
    table = run.table("binning")
    iv = run.table("iv")
    if table is None or table.empty or iv is None or iv.empty:
        return None
    features = iv.head(6)["feature"].tolist()
    if not features:
        return None
    columns = min(3, len(features))
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.6 * rows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        group = table[table["feature"] == feature].sort_values("bin")
        ax.bar(range(len(group)), group["woe"], color=np.where(group["woe"] > 0, "#e45756", "#54a24b"))
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(group["label"], rotation=45, ha="right", fontsize=6)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(feature, fontsize=9)
        ax.set_ylabel("WOE")
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    figure.suptitle("Risk trend by bin (higher WOE = riskier)", y=1.01)
    figure.tight_layout()
    return figure
