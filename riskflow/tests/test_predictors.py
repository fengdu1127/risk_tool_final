"""Parity tests: the exported predictors must reproduce the training libraries.

These are the tests that make the JSON bundle trustworthy. If any of them fail,
production scores have silently diverged from what was measured at training
time, and no other test in the suite would notice.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from riskflow.models.export import (
    ensemble_from_hist_gradient_boosting,
    ensemble_from_xgboost,
    isotonic_from_sklearn,
    linear_from_sklearn,
)
from riskflow.models.predictors import (
    IsotonicCurve,
    LinearScorer,
    Tree,
    TreeEnsemble,
    predictor_from_dict,
)

TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def training_data():
    rng = np.random.default_rng(11)
    n, p = 3000, 6
    X = rng.normal(size=(n, p))
    X[rng.random((n, p)) < 0.08] = np.nan  # missing values on purpose
    weights = np.array([0.9, -0.7, 0.5, 0.0, 0.3, -0.4])
    logit = np.nan_to_num(X) @ weights
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    columns = [f"v{i}" for i in range(p)]
    return pd.DataFrame(X, columns=columns), y, columns


def test_linear_export_matches_sklearn_with_scaling_folded_in(training_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y, columns = training_data
    filled = X.fillna(0.0)
    scaler = StandardScaler().fit(filled.to_numpy())
    estimator = LogisticRegression(max_iter=1000).fit(scaler.transform(filled.to_numpy()), y)

    exported = linear_from_sklearn(estimator, columns, scaler.mean_, scaler.scale_)
    expected = estimator.predict_proba(scaler.transform(filled.to_numpy()))[:, 1]
    np.testing.assert_allclose(exported.predict_proba(filled), expected, atol=TOLERANCE)


def test_linear_scorer_refuses_unencoded_input(training_data):
    X, _, columns = training_data
    scorer = LinearScorer(features=tuple(columns), coefficients=(1.0,) * len(columns), intercept=0.0)
    with pytest.raises(ValueError, match="NaN"):
        scorer.predict_proba(X)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"max_depth": 7, "n_estimators": 120},
        {"monotone_constraints": (1, -1, 1, 0, 0, 0), "scale_pos_weight": 3.0},
        {"n_estimators": 1, "max_depth": 1},
    ],
    ids=["default", "deep", "monotone-and-weighted", "single-stump"],
)
def test_xgboost_export_matches_the_booster(training_data, overrides):
    xgb = pytest.importorskip("xgboost")
    X, y, columns = training_data

    params = dict(n_estimators=40, max_depth=4, learning_rate=0.15, random_state=0, eval_metric="logloss")
    params.update(overrides)
    model = xgb.XGBClassifier(**params).fit(X.to_numpy(), y)

    exported = ensemble_from_xgboost(model, columns)
    expected = model.predict_proba(X.to_numpy())[:, 1]
    np.testing.assert_allclose(exported.predict_proba(X), expected, atol=TOLERANCE)


def test_xgboost_export_keeps_the_missing_direction(training_data):
    """A NaN must follow the branch the booster learned, not a default guess."""
    xgb = pytest.importorskip("xgboost")
    X, y, columns = training_data
    model = xgb.XGBClassifier(n_estimators=25, max_depth=3, random_state=0, eval_metric="logloss")
    model.fit(X.to_numpy(), y)
    exported = ensemble_from_xgboost(model, columns)

    all_missing = pd.DataFrame(np.full((5, len(columns)), np.nan), columns=columns)
    np.testing.assert_allclose(
        exported.predict_proba(all_missing),
        model.predict_proba(all_missing.to_numpy())[:, 1],
        atol=TOLERANCE,
    )


def test_hist_gradient_boosting_export_matches_sklearn(training_data):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y, columns = training_data
    model = HistGradientBoostingClassifier(max_iter=40, max_depth=4, random_state=0).fit(X.to_numpy(), y)

    exported = ensemble_from_hist_gradient_boosting(model, columns)
    expected = model.predict_proba(X.to_numpy())[:, 1]
    np.testing.assert_allclose(exported.predict_proba(X), expected, atol=TOLERANCE)


def test_isotonic_export_matches_sklearn_including_out_of_range(training_data):
    from sklearn.isotonic import IsotonicRegression

    _, y, _ = training_data
    rng = np.random.default_rng(12)
    scores = np.clip(y * 0.3 + rng.normal(0.3, 0.15, len(y)), 0, 1)
    estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y)

    exported = isotonic_from_sklearn(estimator)
    probe = np.concatenate([scores, [scores.min() - 5, scores.max() + 5]])
    np.testing.assert_allclose(exported.predict(probe), estimator.predict(probe), atol=1e-8)


@pytest.mark.parametrize("build", ["linear", "xgboost"])
def test_predictors_round_trip_through_json(training_data, build):
    X, y, columns = training_data
    if build == "linear":
        original = LinearScorer(features=tuple(columns), coefficients=tuple(np.linspace(-1, 1, len(columns))), intercept=-0.5)
        frame = X.fillna(0.0)
    else:
        xgb = pytest.importorskip("xgboost")
        model = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=0, eval_metric="logloss").fit(X.to_numpy(), y)
        original = ensemble_from_xgboost(model, columns)
        frame = X

    restored = predictor_from_dict(json.loads(json.dumps(original.to_dict())))
    np.testing.assert_array_equal(restored.predict_proba(frame), original.predict_proba(frame))


def test_split_convention_is_honoured_at_the_boundary():
    """`lt` and `le` must genuinely differ for a value sitting on the threshold."""
    tree = Tree(
        feature=np.array([0, -1, -1]),
        threshold=np.array([1.0, 0.0, 0.0]),
        left=np.array([1, 1, 2]),
        right=np.array([2, 1, 2]),
        missing=np.array([1, 1, 2]),
        value=np.array([0.0, -3.0, 3.0]),
    )
    frame = pd.DataFrame({"x": [1.0]})
    strict = TreeEnsemble(features=("x",), trees=(tree,), base_margin=0.0, split_op="lt")
    inclusive = TreeEnsemble(features=("x",), trees=(tree,), base_margin=0.0, split_op="le")

    assert strict.margin(frame)[0] == 3.0  # 1.0 < 1.0 is false, so it goes right
    assert inclusive.margin(frame)[0] == -3.0  # 1.0 <= 1.0 is true, so it goes left


def test_missing_inputs_are_reported_by_name():
    scorer = LinearScorer(features=("a", "b"), coefficients=(1.0, 1.0), intercept=0.0)
    with pytest.raises(KeyError, match="'b'"):
        scorer.predict_proba(pd.DataFrame({"a": [1.0]}))


def test_sigmoid_stays_finite_at_extreme_margins():
    scorer = LinearScorer(features=("a",), coefficients=(1.0,), intercept=0.0)
    probabilities = scorer.predict_proba(pd.DataFrame({"a": [-800.0, 0.0, 800.0]}))
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] == pytest.approx(0.0, abs=1e-12)
    assert probabilities[2] == pytest.approx(1.0, abs=1e-12)


def test_empty_calibration_curve_is_an_identity():
    assert IsotonicCurve(x=(), y=()).predict([0.2, 0.8]).tolist() == [0.2, 0.8]
