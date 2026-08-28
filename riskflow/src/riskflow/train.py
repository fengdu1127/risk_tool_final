"""End-to-end training: data in, promoted-ready run directory out.

The order below is the whole method, and each step is deliberately narrow enough
to test on its own:

    split → bin → screen → model → calibrate → mine rules → set cutoffs
    → assemble bundle → replay decisions → write the run

The last two steps matter as much as the modelling ones. The bundle is assembled
from the exported artefacts and then used to re-score every sample, so the
numbers in the report come from the same code that will run in production rather
than from the in-memory objects that trained it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .bundle import ScoringBundle
from .data.schema import DatasetSchema, coerce_label, infer_schema, validate_frame
from .data.splitting import split
from .features.diagnostics import screen
from .features.woe import WoeTransformer
from .logging_setup import get_logger, run_log_file
from .models.training import train_models
from .monitoring.drift import DriftBaseline
from .policy.decision import DecisionPolicy
from .policy.mining import mine_rules
from .policy.thresholds import search_global_cutoff, search_segment_cutoffs
from .policy.validation import backtest, combined_effect, marginal_contribution, select_stable, stability
from .registry import Registry, Run
from .settings import Settings

log = get_logger("train")


@dataclass
class TrainingRun:
    run: Run
    bundle: ScoringBundle
    summary: dict


def train(
    data: str | Path | pd.DataFrame,
    label: str,
    settings: Settings | None = None,
    features: Sequence[str] | None = None,
    time_col: str | None = None,
    id_col: str | None = None,
    run_name: str | None = None,
    registry_root: str | Path | None = None,
) -> TrainingRun:
    settings = settings or Settings()
    if time_col:
        settings = settings.merged({"split": {"time_col": time_col}})

    registry = Registry(registry_root or settings.report.output_root)
    run = registry.new_run(run_name)

    with run_log_file(run.path / "run.log"):
        try:
            return _train(data, label, settings, features, id_col, run)
        except Exception:
            log.exception("training failed; the partial run is left at %s for inspection", run.path)
            raise


def _train(
    data: str | Path | pd.DataFrame,
    label: str,
    settings: Settings,
    features: Sequence[str] | None,
    id_col: str | None,
    run: Run,
) -> TrainingRun:
    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    source = "<dataframe>" if isinstance(data, pd.DataFrame) else str(data)
    log.info("training on %s | %d rows x %d columns", source, len(df), df.shape[1])

    df = coerce_label(df, label)
    schema = infer_schema(df, label, features, settings.split.time_col, id_col)
    for message in validate_frame(df, schema):
        log.warning("input check: %s", message)
    if df[label].isna().any():
        df = df.loc[df[label].notna()].reset_index(drop=True)

    # --- split -------------------------------------------------------------
    parts = split(df, label, settings.split)
    datasets = {name: frame for name, frame in parts.items()}
    run.write_table("split_profile", parts.profile(label, settings.split.time_col))

    # --- bin and screen ----------------------------------------------------
    woe_all = WoeTransformer.fit(parts.train, schema, settings.binning)
    screening = screen(parts.train, parts.test, schema, woe_all, settings.screening)
    woe = woe_all.subset(screening.selected)
    model_schema = schema.subset(screening.selected)
    run.write_table("screening", screening.report)
    run.write_table("binning", woe.bin_table())
    run.write_table("iv", woe.iv_table())
    if len(screening.correlated):
        run.write_table("correlations", screening.correlated)

    # --- model -------------------------------------------------------------
    models = train_models(datasets, schema, woe, screening.selected, settings.model)
    run.write_table("model_metrics", models.metrics)
    run.write_table("gains", models.gains)
    run.write_table("calibration", models.calibration)
    if models.scorecard is not None:
        run.write_table("scorecard", models.scorecard)

    scores = {name: models.best.scores[name] for name in datasets}

    # --- rules -------------------------------------------------------------
    candidates = mine_rules(parts.train, model_schema, screening.selected, settings.rules)
    rule_backtest = backtest(candidates, datasets, label)
    rule_stability = stability(rule_backtest, settings.rules)
    stable_rules = select_stable(candidates, rule_stability)
    if len(rule_backtest):
        run.write_table("rules_backtest", rule_backtest)
        run.write_table("rules_stability", rule_stability)
    if stable_rules:
        run.write_table("rules_combined", combined_effect(stable_rules, datasets, label))
        run.write_table("rules_marginal", marginal_contribution(stable_rules, parts.test, label))

    # --- cutoffs -----------------------------------------------------------
    cutoff_search = search_global_cutoff(datasets, scores, label, settings.cutoffs)
    run.write_table("cutoff_candidates", cutoff_search.candidates)
    run.write_table("cutoff_performance", cutoff_search.performance)

    segment_cutoffs, segment_report = search_segment_cutoffs(
        datasets, scores, label, cutoff_search.chosen, settings.cutoffs
    )
    if len(segment_report):
        run.write_table("segments", segment_report)

    policy = DecisionPolicy(
        global_cutoff=cutoff_search.chosen,
        reject_rules=tuple(stable_rules),
        segment_cutoffs=tuple(segment_cutoffs),
    )

    # --- assemble ----------------------------------------------------------
    drift = DriftBaseline.fit(parts.train, woe, screening.selected, scores["train"])
    bundle = ScoringBundle(
        schema=model_schema,
        woe=woe,
        space=models.best.space,
        predictor=models.best.predictor,
        policy=policy,
        drift=drift,
        calibrator=models.calibrator,
        scorecard_settings=settings.model.scorecard,
        metadata={
            "run": run.name,
            "source": source,
            "algorithm": models.best_name,
            "backend": models.best.backend,
            "hyperparameters": models.best.params,
            "split_strategy": parts.strategy,
            "metrics": models.metrics.to_dict(orient="records"),
            "diagnostics": models.diagnostics,
        },
    )
    bundle.save(run.bundle_path)

    # --- replay through the deployed path ----------------------------------
    decisions = _replay(bundle, datasets, label, models, run)

    summary = _summarise(
        run, bundle, parts, screening, models, candidates, stable_rules,
        cutoff_search, segment_cutoffs, decisions, settings, label,
    )
    run.summary_path.write_text(_dumps(summary), encoding="utf-8")
    settings.to_json(run.path / "settings.json")

    _write_report(run, settings)
    log.info("run complete: %s", run.path)
    return TrainingRun(run=run, bundle=bundle, summary=summary)


def _replay(bundle: ScoringBundle, datasets, label: str, models, run: Run) -> pd.DataFrame:
    """Re-score every sample through the saved bundle and check it agrees.

    This is the train/serve consistency gate: if the exported artefacts disagree
    with the in-memory model by more than rounding, the run fails here rather
    than in production.
    """
    frames = []
    for name, frame in datasets.items():
        scored = bundle.score(frame)
        drift_max = float(np.max(np.abs(scored["model_score"].to_numpy() - models.best.scores[name])))
        if drift_max > 1e-6:
            raise AssertionError(
                f"the saved bundle scores {name} differently from the fitted model "
                f"(max difference {drift_max:.2e}); export is not faithful"
            )
        summary = bundle.policy.summarise(frame, scored["model_score"], label)
        summary.insert(0, "dataset", name)
        frames.append(summary)
    log.info("bundle replay matches the fitted model on every sample")
    decisions = pd.concat(frames, ignore_index=True)
    run.write_table("decisions", decisions)
    return decisions


def _summarise(
    run, bundle, parts, screening, models, candidates, stable_rules,
    cutoff_search, segment_cutoffs, decisions, settings, label,
) -> dict:
    holdout_reject = decisions[(decisions["dataset"] == "holdout") & (decisions["action"] == "reject")]
    holdout_approve = decisions[(decisions["dataset"] == "holdout") & (decisions["action"] == "approve")]
    return {
        "run": run.name,
        "label": label,
        "split": {
            "strategy": parts.strategy,
            "detail": parts.detail,
            "rows": {name: len(frame) for name, frame in parts.items()},
        },
        "features": {
            "considered": len(screening.report),
            "selected": list(screening.selected),
            "dropped": int((~screening.report["selected"]).sum()),
        },
        "model": {
            "algorithm": models.best_name,
            "backend": models.best.backend,
            "metrics": models.metrics.to_dict(orient="records"),
            "diagnostics": models.diagnostics,
            "calibrated": bundle.calibrator is not None,
        },
        "rules": {
            "mined": len(candidates),
            "stable": len(stable_rules),
            "descriptions": [rule.describe() for rule in stable_rules],
        },
        "policy": {
            "global_cutoff": cutoff_search.chosen.to_dict(),
            "segment_overrides": [s.to_dict() for s in segment_cutoffs],
        },
        "outcome_on_holdout": {
            "reject_rate": float(holdout_reject["share"].iloc[0]) if len(holdout_reject) else None,
            "rejected_bad_rate": float(holdout_reject["bad_rate"].iloc[0]) if len(holdout_reject) else None,
            "approved_bad_rate": float(holdout_approve["bad_rate"].iloc[0]) if len(holdout_approve) else None,
            "bad_capture_at_reject": float(holdout_reject["bad_capture"].iloc[0]) if len(holdout_reject) else None,
        },
    }


def _write_report(run: Run, settings: Settings) -> None:
    try:
        from .reporting.html import write_report

        write_report(run, make_plots=settings.report.make_plots, dpi=settings.report.fig_dpi)
    except Exception as exc:
        log.warning("report generation failed (the run itself is fine): %s", exc)


def _dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)
