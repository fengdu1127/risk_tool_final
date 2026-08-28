"""Tests for the methodological claims the training module makes.

Each of these guards a promise that is easy to break silently and impossible to
notice from accuracy numbers alone.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from riskflow.features.diagnostics import screen
from riskflow.features.space import mixed_space, woe_space
from riskflow.models import training as T
from riskflow.models.predictors import LinearScorer, TreeEnsemble
from riskflow.models.scorecard import build_scorecard, to_credit_score, verify_scorecard
from riskflow.settings import ModelSettings, ScorecardSettings, Settings

LABEL = "is_bad"


@pytest.fixture(scope="module")
def fitted(applications, schema, settings):
    from riskflow.data.splitting import split
    from riskflow.features.woe import WoeTransformer

    parts = split(applications, LABEL, settings.split)
    woe = WoeTransformer.fit(parts.train, schema, settings.binning)
    screening = screen(parts.train, parts.test, schema, woe, settings.screening)
    trimmed = woe.subset(screening.selected)
    datasets = {name: frame for name, frame in parts.items()}
    result = T.train_models(datasets, schema, trimmed, screening.selected, settings.model)
    return result, datasets, trimmed, screening.selected


# ------------------------------------------------------------------- no leakage


def test_tuning_cannot_reach_the_test_set():
    """A structural guarantee, checked structurally: test is not a parameter."""
    parameters = set(inspect.signature(T._tune).parameters)
    assert "X_test" not in parameters and "test" not in parameters
    assert parameters == {"algorithm", "X", "y", "settings", "constraints"}


def test_calibration_uses_out_of_fold_train_predictions_not_test(fitted, monkeypatch):
    result, datasets, woe, _ = fitted
    seen: list[int] = []
    original = T._fit_and_export

    def spy(algorithm, X, y, params, settings, constraints):
        seen.append(len(X))
        return original(algorithm, X, y, params, settings, constraints)

    monkeypatch.setattr(T, "_fit_and_export", spy)
    T._fit_calibrator(result.best, datasets["train"], datasets["train"][LABEL].to_numpy(float), woe, ModelSettings(cv_folds=3, search_iterations=0), {})

    assert seen, "calibration should refit the model on folds"
    # every refit sees a subset of train, never the test or holdout row counts
    assert all(size < len(datasets["train"]) for size in seen)
    assert all(size != len(datasets["test"]) for size in seen)


def test_calibration_improves_probability_accuracy(fitted):
    result, datasets, _, _ = fitted
    if result.calibrator is None:
        pytest.skip("calibration was not fitted for this sample")
    holdout = result.calibration[result.calibration["dataset"] == "holdout"]
    assert holdout["calibrated_error"].mean() <= holdout["raw_error"].mean()


# ------------------------------------------------------------------ constraints


def test_monotone_directions_follow_each_feature_s_risk(fitted):
    _, datasets, woe, features = fitted
    from riskflow.data.schema import infer_schema

    schema = infer_schema(datasets["train"], LABEL, time_col="apply_time", id_col="application_id")
    space = mixed_space(features, schema.numeric)
    directions = T._monotone_directions(woe, space, ModelSettings())

    # WOE columns are risk-ordered by construction
    for column in space.woe_columns:
        assert directions[column] == 1
    # more delinquencies means more risk; more income means less
    if "max_delinquency" in directions:
        assert directions["max_delinquency"] == 1
    if "income" in directions:
        assert directions["income"] == -1


def test_a_monotone_gbdt_never_scores_a_worse_applicant_better(fitted):
    result, datasets, woe, features = fitted
    gbdt = result.candidates.get("gbdt")
    if gbdt is None or not isinstance(gbdt.predictor, TreeEnsemble):
        pytest.skip("no gradient boosted candidate in this run")
    if "max_delinquency" not in gbdt.space.raw_columns:
        pytest.skip("max_delinquency is not a raw input to the ensemble")

    base = datasets["holdout"].head(200)
    scores = []
    for value in (0, 1, 2, 3, 4, 5):
        probe = base.copy()
        probe["max_delinquency"] = value
        scores.append(gbdt.predictor.predict_proba(gbdt.space.build(probe, woe)))
    stacked = np.vstack(scores)
    # raising only the delinquency count must never lower the risk estimate
    assert np.all(np.diff(stacked, axis=0) >= -1e-9)


# ------------------------------------------------------------------- scorecard


def test_scorecard_points_reconcile_with_the_model(fitted):
    result, datasets, woe, _ = fitted
    logistic = result.candidates.get("logistic")
    if logistic is None:
        pytest.skip("no logistic candidate in this run")

    card = build_scorecard(logistic.predictor, woe, ScorecardSettings())
    gap = verify_scorecard(card, logistic.predictor, woe, datasets["holdout"], ScorecardSettings())
    assert gap < 1.0, f"the printed card disagrees with the model by {gap:.2f} points"


def test_riskier_bins_earn_fewer_points(fitted):
    result, _, woe, _ = fitted
    logistic = result.candidates.get("logistic")
    if logistic is None:
        pytest.skip("no logistic candidate in this run")

    card = build_scorecard(logistic.predictor, woe, ScorecardSettings())
    body = card[card["feature"] != "__base__"]
    for feature, group in body.groupby("feature"):
        if len(group) < 3 or group["coefficient"].iloc[0] <= 0:
            continue
        correlation = group["bad_rate"].corr(group["points"])
        assert correlation < 0, f"points for '{feature}' should fall as risk rises"


def test_the_credit_score_scale_behaves_as_advertised():
    settings = ScorecardSettings(pdo=20, base_score=600, base_odds=1 / 15)
    at_base = to_credit_score([1 / 16], settings)[0]
    assert at_base == pytest.approx(600.0, abs=1e-6), "base odds should land on the base score"

    # every `pdo` points doubles the good-to-bad odds
    odds = 1 / 15
    doubled = 1 / (1 + 1 / (odds / 2))
    assert to_credit_score([doubled], settings)[0] == pytest.approx(620.0, abs=1e-6)


# ---------------------------------------------------------------- model choice


def test_selection_happens_on_test_and_reports_all_three_samples(fitted):
    result, _, _, _ = fitted
    assert set(result.metrics["dataset"]) == {"train", "test", "holdout"}
    on_test = result.metrics[result.metrics["dataset"] == "test"]
    assert result.best_name == on_test.loc[on_test["ks"].idxmax(), "model"]


def test_overfit_diagnostics_are_reported(fitted):
    result, _, _, _ = fitted
    assert "train_test_ks_gap" in result.diagnostics
    assert result.diagnostics["overfit_verdict"] in {"acceptable", "moderate", "severe"}


def test_an_unknown_algorithm_is_rejected(fitted):
    _, datasets, woe, features = fitted
    from riskflow.data.schema import infer_schema

    schema = infer_schema(datasets["train"], LABEL, time_col="apply_time", id_col="application_id")
    with pytest.raises(ValueError, match="unknown algorithm"):
        T.train_models(datasets, schema, woe, features, ModelSettings(algorithms=("random_forest",)))


# ------------------------------------------------------------- feature spaces


def test_feature_spaces_build_the_matrix_the_model_expects(fitted):
    _, datasets, woe, features = fitted
    from riskflow.data.schema import infer_schema

    schema = infer_schema(datasets["train"], LABEL, time_col="apply_time", id_col="application_id")
    encoded = woe_space(features).build(datasets["test"], woe)
    mixed = mixed_space(features, schema.numeric).build(datasets["test"], woe)

    assert list(encoded.columns) == list(features)
    assert not encoded.isna().any().any(), "the WOE space must be complete"
    # the mixed space keeps raw numerics, missing values and all
    assert set(mixed.columns) == set(features)
    assert mixed[[c for c in mixed.columns if c in schema.numeric]].isna().any().any()


def test_a_feature_space_names_what_it_is_missing(fitted):
    _, datasets, woe, features = fitted
    space = woe_space(features)
    with pytest.raises(KeyError, match=features[0]):
        space.build(datasets["test"].drop(columns=[features[0]]), woe)
