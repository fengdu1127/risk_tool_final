"""Convert fitted scikit-learn / XGBoost objects into serialisable predictors.

Every converter here is covered by a parity test that asserts the exported
predictor reproduces the original library's own `predict_proba` to within 1e-6.
That test is the contract: if a library changes its internals, the test fails
loudly instead of production scores drifting quietly.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .predictors import IsotonicCurve, LinearScorer, Tree, TreeEnsemble


def linear_from_sklearn(
    estimator,
    features: Sequence[str],
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> LinearScorer:
    """Export a fitted LogisticRegression, folding any standardisation into the weights."""
    coef = np.asarray(estimator.coef_, dtype=float).ravel()
    intercept = float(np.asarray(estimator.intercept_, dtype=float).ravel()[0])
    if len(coef) != len(features):
        raise ValueError(f"model has {len(coef)} coefficients but {len(features)} feature names")
    if mean is not None and scale is not None:
        scale = np.where(np.asarray(scale, dtype=float) == 0, 1.0, np.asarray(scale, dtype=float))
        mean = np.asarray(mean, dtype=float)
        intercept = intercept - float(np.sum(coef * mean / scale))
        coef = coef / scale
    return LinearScorer(
        features=tuple(features),
        coefficients=tuple(float(c) for c in coef),
        intercept=intercept,
    )


def isotonic_from_sklearn(estimator) -> IsotonicCurve:
    """Export a fitted IsotonicRegression as an interpolation table."""
    x = np.asarray(estimator.X_thresholds_, dtype=float)
    y = np.asarray(estimator.y_thresholds_, dtype=float)
    return IsotonicCurve(x=tuple(float(v) for v in x), y=tuple(float(v) for v in y))


def ensemble_from_xgboost(model, features: Sequence[str]) -> TreeEnsemble:
    """Export a fitted XGBClassifier (binary:logistic) into a TreeEnsemble.

    Reads the booster's native JSON rather than its text dump: the dump rounds
    thresholds and leaf weights when printing, which silently reroutes rows that
    sit on a split boundary.
    """
    booster = model.get_booster()
    raw = json.loads(bytearray(booster.save_raw(raw_format="json")).decode("utf-8"))
    learner = raw["learner"]
    objective = learner["objective"]["name"]
    if objective != "binary:logistic":
        raise ValueError(f"only binary:logistic is supported for export, got '{objective}'")
    if int(learner["learner_model_param"].get("num_class", 0) or 0) > 1:
        raise ValueError("multi-class boosters are not supported for export")

    # XGBoost stores base_score in probability space for logistic objectives; the
    # additive margin starts from its logit.
    base_score = _parse_base_score(learner["learner_model_param"]["base_score"])
    base_score = min(max(base_score, 1e-9), 1 - 1e-9)
    base_margin = float(np.log(base_score / (1.0 - base_score)))

    raw_trees = learner["gradient_booster"]["model"]["trees"]
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None and 0 <= best_iteration + 1 < len(raw_trees):
        # Early stopping makes predict() use only the trees up to best_iteration.
        raw_trees = raw_trees[: best_iteration + 1]

    trees = [_tree_from_xgb_json(t, len(features)) for t in raw_trees]
    return TreeEnsemble(
        features=tuple(features),
        trees=tuple(trees),
        base_margin=base_margin,
        split_op="lt",
        dtype="float32",
    )


def _tree_from_xgb_json(tree: dict, n_features: int) -> Tree:
    if any(int(t) != 0 for t in tree.get("split_type", [])):
        raise ValueError("categorical splits are not supported for export; encode them first")

    left = np.asarray(tree["left_children"], dtype=int)
    right = np.asarray(tree["right_children"], dtype=int)
    conditions = np.asarray(tree["split_conditions"], dtype=float)
    indices = np.asarray(tree["split_indices"], dtype=int)
    default_left = np.asarray(tree["default_left"], dtype=bool)

    is_leaf = left < 0
    if (indices[~is_leaf] >= n_features).any():
        raise ValueError("booster references a feature index outside the schema")

    self_index = np.arange(len(left), dtype=int)
    feature = np.where(is_leaf, -1, indices).astype(int)
    # For a leaf, split_conditions holds the leaf's output weight.
    value = np.where(is_leaf, conditions, 0.0).astype(float)
    threshold = np.where(is_leaf, 0.0, conditions).astype(float)
    left_child = np.where(is_leaf, self_index, left).astype(int)
    right_child = np.where(is_leaf, self_index, right).astype(int)
    missing = np.where(is_leaf, self_index, np.where(default_left, left, right)).astype(int)
    return Tree(
        feature=feature,
        threshold=threshold,
        left=left_child,
        right=right_child,
        missing=missing,
        value=value,
    )


def _parse_base_score(raw) -> float:
    """XGBoost reports base_score as a bare number or a bracketed vector string."""
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().strip("[]")
    first = text.split(",")[0].strip()
    if not first:
        raise ValueError(f"cannot parse base_score from {raw!r}")
    return float(first)


def ensemble_from_hist_gradient_boosting(model, features: Sequence[str]) -> TreeEnsemble:
    """Export a fitted HistGradientBoostingClassifier into a TreeEnsemble."""
    baseline = float(np.asarray(model._baseline_prediction, dtype=float).ravel()[0])
    trees = []
    for stage in model._predictors:
        if len(stage) != 1:
            raise ValueError("only binary classification is supported for export")
        trees.append(_tree_from_hist_predictor(stage[0]))
    return TreeEnsemble(
        features=tuple(features), trees=tuple(trees), base_margin=baseline, split_op="le"
    )


def _tree_from_hist_predictor(predictor) -> Tree:
    nodes = predictor.nodes
    size = len(nodes)
    if "is_categorical" in nodes.dtype.names and nodes["is_categorical"].any():
        raise ValueError("categorical splits are not supported for export; encode them first")

    is_leaf = nodes["is_leaf"].astype(bool)
    feature = np.where(is_leaf, -1, nodes["feature_idx"]).astype(int)
    threshold = np.asarray(nodes["num_threshold"], dtype=float)
    left = np.asarray(nodes["left"], dtype=int)
    right = np.asarray(nodes["right"], dtype=int)
    value = np.where(is_leaf, nodes["value"], 0.0).astype(float)
    missing_left = np.asarray(nodes["missing_go_to_left"], dtype=bool)
    missing = np.where(missing_left, left, right).astype(int)

    # Leaves must point at themselves so a malformed walk cannot wander.
    self_index = np.arange(size, dtype=int)
    left = np.where(is_leaf, self_index, left)
    right = np.where(is_leaf, self_index, right)
    missing = np.where(is_leaf, self_index, missing)
    return Tree(feature=feature, threshold=threshold, left=left, right=right, missing=missing, value=value)
