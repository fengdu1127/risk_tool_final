"""Model training: candidate fitting, tuning, selection and calibration.

Three rules shape this module:

1. Tuning never sees the test set. Hyper-parameters are chosen on an inner split
   carved out of train, so test stays an honest model-selection sample.
2. Calibration never sees the test set either. The isotonic curve is fitted on
   out-of-fold predictions over train, so selecting on test and calibrating on
   test cannot compound into the same optimistic bias.
3. The holdout is opened once, at the end, and only to report.

Every fitted model is immediately exported to a numpy-only predictor, so the
object measured during training is byte-for-byte the object that scores in
production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.schema import DatasetSchema
from ..features.space import FeatureSpace, mixed_space, woe_space
from ..features.woe import WoeTransformer, monotonic_correlation
from ..logging_setup import get_logger
from ..settings import ModelSettings
from . import metrics as M
from .export import (
    ensemble_from_hist_gradient_boosting,
    ensemble_from_xgboost,
    isotonic_from_sklearn,
    linear_from_sklearn,
)
from .predictors import IsotonicCurve, LinearScorer, Predictor
from .scorecard import build_scorecard, verify_scorecard

log = get_logger("model")

DATASETS = ("train", "test", "holdout")


@dataclass
class Candidate:
    name: str
    predictor: Predictor
    space: FeatureSpace
    params: dict
    backend: str
    scores: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class TrainingResult:
    candidates: dict[str, Candidate]
    best_name: str
    calibrator: IsotonicCurve | None
    metrics: pd.DataFrame
    gains: pd.DataFrame
    calibration: pd.DataFrame
    scorecard: pd.DataFrame | None
    diagnostics: dict
    band_edges: np.ndarray

    @property
    def best(self) -> Candidate:
        return self.candidates[self.best_name]

    def scored(self, dataset: str) -> np.ndarray:
        return self.best.scores[dataset]


def train_models(
    datasets: Mapping[str, pd.DataFrame],
    schema: DatasetSchema,
    woe: WoeTransformer,
    features: Sequence[str],
    settings: ModelSettings,
) -> TrainingResult:
    train, test = datasets["train"], datasets["test"]
    y_train = train[schema.label].to_numpy(dtype=float)

    spaces = {
        "logistic": woe_space(features),
        "gbdt": mixed_space(features, schema.numeric),
    }
    constraints = _monotone_directions(woe, spaces["gbdt"], settings)

    candidates: dict[str, Candidate] = {}
    for algorithm in settings.algorithms:
        if algorithm not in spaces:
            raise ValueError(f"unknown algorithm '{algorithm}'; expected one of {sorted(spaces)}")
        space = spaces[algorithm]
        X_train = space.build(train, woe)
        params = _tune(algorithm, X_train, y_train, settings, constraints)
        predictor, backend = _fit_and_export(algorithm, X_train, y_train, params, settings, constraints)
        candidate = Candidate(name=algorithm, predictor=predictor, space=space, params=params, backend=backend)
        for name, frame in datasets.items():
            candidate.scores[name] = predictor.predict_proba(space.build(frame, woe))
        candidates[algorithm] = candidate
        log.info("fitted %s (%s) with %d input(s)", algorithm, backend, len(space.columns))

    metrics = pd.DataFrame(
        [
            {"model": name, **M.summary(datasets[ds][schema.label], candidate.scores[ds], ds)}
            for name, candidate in candidates.items()
            for ds in DATASETS
            if ds in datasets
        ]
    )
    best_name = _select_best(metrics, settings)
    best = candidates[best_name]
    log.info("selected '%s' on test %s", best_name, settings.primary_metric)

    calibrator = None
    if settings.calibration == "isotonic":
        calibrator = _fit_calibrator(
            best, train, y_train, woe, settings, constraints
        )

    band_edges = M.band_edges(best.scores["train"], 10)
    gains = pd.concat(
        [
            M.gains_table(datasets[ds][schema.label], best.scores[ds], ds, edges=band_edges)
            for ds in DATASETS
            if ds in datasets
        ],
        ignore_index=True,
    )
    calibration = _calibration_report(datasets, schema.label, best, calibrator)

    scorecard = None
    if "logistic" in candidates and isinstance(candidates["logistic"].predictor, LinearScorer):
        scorer = candidates["logistic"].predictor
        scorecard = build_scorecard(scorer, woe, settings.scorecard)
        gap = verify_scorecard(scorecard, scorer, woe, train, settings.scorecard)
        if gap > 1.0:
            raise AssertionError(f"scorecard points disagree with the model by {gap:.2f} points")
        log.info("scorecard reconciles with the model to within %.3f points", gap)

    return TrainingResult(
        candidates=candidates,
        best_name=best_name,
        calibrator=calibrator,
        metrics=metrics,
        gains=gains,
        calibration=calibration,
        scorecard=scorecard,
        diagnostics=_diagnostics(metrics, best_name),
        band_edges=band_edges,
    )


# --------------------------------------------------------------------------- #
# fitting backends
# --------------------------------------------------------------------------- #


def _fit_and_export(
    algorithm: str,
    X: pd.DataFrame,
    y: np.ndarray,
    params: Mapping,
    settings: ModelSettings,
    constraints: Mapping[str, int],
) -> tuple[Predictor, str]:
    if algorithm == "logistic":
        return _fit_logistic(X, y, params, settings), "sklearn.LogisticRegression"
    if algorithm == "gbdt":
        return _fit_gbdt(X, y, params, settings, constraints)
    raise ValueError(f"unknown algorithm '{algorithm}'")


def _fit_logistic(X: pd.DataFrame, y: np.ndarray, params: Mapping, settings: ModelSettings) -> LinearScorer:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    values = X.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("logistic input contains NaN; the WOE space should be complete")
    scaler = StandardScaler().fit(values)
    estimator = LogisticRegression(
        C=float(params.get("C", 1.0)),
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced" if settings.balance_classes else None,
        random_state=settings.random_state,
    ).fit(scaler.transform(values), y)
    # Standardisation is folded into the exported weights, so the deployed model
    # is a plain dot product with no scaler to keep in sync.
    return linear_from_sklearn(estimator, X.columns, scaler.mean_, scaler.scale_)


def _fit_gbdt(
    X: pd.DataFrame,
    y: np.ndarray,
    params: Mapping,
    settings: ModelSettings,
    constraints: Mapping[str, int],
) -> tuple[Predictor, str]:
    directions = tuple(int(constraints.get(c, 0)) for c in X.columns) if settings.monotone_constraints else None
    values = X.to_numpy(dtype=float)
    positives = float(y.sum())
    scale_pos_weight = ((len(y) - positives) / positives) if (settings.balance_classes and positives) else 1.0

    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 5)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            gamma=float(params.get("gamma", 0.0)),
            scale_pos_weight=scale_pos_weight,
            monotone_constraints=directions,
            random_state=settings.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        # Fitting on a bare matrix keeps the export independent of column naming.
        model.fit(values, y, verbose=False)
        return ensemble_from_xgboost(model, X.columns), "xgboost.XGBClassifier"
    except ImportError:
        pass

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        min_samples_leaf=int(params.get("min_child_weight", 5)) * 4,
        l2_regularization=float(params.get("reg_lambda", 1.0)),
        monotonic_cst=list(directions) if directions else None,
        random_state=settings.random_state,
        early_stopping=False,
    ).fit(values, y)
    return ensemble_from_hist_gradient_boosting(model, X.columns), "sklearn.HistGradientBoostingClassifier"


def _monotone_directions(
    woe: WoeTransformer, space: FeatureSpace, settings: ModelSettings
) -> dict[str, int]:
    """Per-column risk direction for the GBDT's monotone constraints.

    WOE columns are +1 by construction (higher WOE means higher risk). Raw
    numeric columns take the direction their own binning already established,
    which respects missing values in a way a plain correlation would not.
    """
    directions: dict[str, int] = {c: 1 for c in space.woe_columns}
    for column in space.raw_columns:
        binning = woe.binnings.get(column)
        correlation = monotonic_correlation(binning) if binning else float("nan")
        if np.isnan(correlation) or abs(correlation) < settings.monotone_min_abs_corr:
            directions[column] = 0
        else:
            directions[column] = 1 if correlation > 0 else -1
    return directions


# --------------------------------------------------------------------------- #
# tuning, selection, calibration
# --------------------------------------------------------------------------- #


_SEARCH_SPACE: dict[str, dict[str, Callable[[np.random.Generator], float]]] = {
    "logistic": {
        "C": lambda rng: float(10 ** rng.uniform(-3, 1)),
    },
    "gbdt": {
        "n_estimators": lambda rng: int(rng.integers(80, 400)),
        "max_depth": lambda rng: int(rng.integers(2, 6)),
        "learning_rate": lambda rng: float(10 ** rng.uniform(-2, -0.7)),
        "subsample": lambda rng: float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": lambda rng: float(rng.uniform(0.6, 1.0)),
        "min_child_weight": lambda rng: int(rng.integers(1, 30)),
        "reg_lambda": lambda rng: float(10 ** rng.uniform(-1, 1.5)),
        "gamma": lambda rng: float(rng.uniform(0, 3)),
    },
}


def _tune(
    algorithm: str,
    X: pd.DataFrame,
    y: np.ndarray,
    settings: ModelSettings,
    constraints: Mapping[str, int] | None,
) -> dict:
    """Random search scored on a split carved out of TRAIN.

    The test frame is not a parameter here, which is the point: tuning has no way
    to reach it, so test remains an untouched model-selection sample.
    """
    space = _SEARCH_SPACE[algorithm]
    if settings.search_iterations <= 0:
        return {name: draw(np.random.default_rng(settings.random_state)) for name, draw in space.items()}

    rng = np.random.default_rng(settings.random_state)
    inner_train, inner_valid = _stratified_indices(y, 0.8, rng)
    X_fit, y_fit = X.iloc[inner_train], y[inner_train]
    X_val, y_val = X.iloc[inner_valid], y[inner_valid]

    best_params, best_score = None, -np.inf
    for _ in range(settings.search_iterations):
        params = {name: draw(rng) for name, draw in space.items()}
        try:
            predictor, _ = _fit_and_export(algorithm, X_fit, y_fit, params, settings, constraints or {})
            score = M.ks(y_val, predictor.predict_proba(X_val))
        except Exception as exc:  # a bad draw must not end the search
            log.debug("%s draw failed (%s): %s", algorithm, params, exc)
            continue
        if np.isfinite(score) and score > best_score:
            best_params, best_score = params, score

    if best_params is None:
        raise RuntimeError(f"every {algorithm} configuration failed to fit")
    log.info("%s tuned over %d draws | inner-validation KS=%.4f", algorithm, settings.search_iterations, best_score)
    return best_params


def _stratified_indices(y: np.ndarray, train_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for value in np.unique(y):
        idx = np.nonzero(y == value)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * train_fraction))
        left.append(idx[:cut])
        right.append(idx[cut:])
    return np.concatenate(left), np.concatenate(right)


def _select_best(metrics: pd.DataFrame, settings: ModelSettings) -> str:
    metric = settings.primary_metric
    if metric not in metrics.columns:
        raise ValueError(f"primary_metric '{metric}' is not a reported metric")
    on_test = metrics[metrics["dataset"] == "test"]
    if on_test.empty:
        raise ValueError("no test-set metrics; cannot select a model")
    return str(on_test.loc[on_test[metric].idxmax(), "model"])


def _fit_calibrator(
    best: Candidate,
    train: pd.DataFrame,
    y_train: np.ndarray,
    woe: WoeTransformer,
    settings: ModelSettings,
    constraints: Mapping[str, int],
) -> IsotonicCurve | None:
    """Isotonic calibration on out-of-fold train predictions.

    Fitting on in-sample scores would learn the model's own overconfidence, and
    fitting on test would spend the sample twice — once to pick the model, once
    to calibrate it. Out-of-fold predictions avoid both.
    """
    from sklearn.isotonic import IsotonicRegression

    X = best.space.build(train, woe)
    folds = max(2, settings.cv_folds)
    rng = np.random.default_rng(settings.random_state)
    assignments = _fold_assignments(y_train, folds, rng)

    oof = np.full(len(y_train), np.nan)
    for fold in range(folds):
        holdout_mask = assignments == fold
        if holdout_mask.all() or not holdout_mask.any():
            continue
        try:
            predictor, _ = _fit_and_export(
                best.name, X.iloc[~holdout_mask], y_train[~holdout_mask], best.params, settings, constraints
            )
        except Exception as exc:
            log.warning("calibration fold %d failed: %s", fold, exc)
            continue
        oof[holdout_mask] = predictor.predict_proba(X.iloc[holdout_mask])

    usable = ~np.isnan(oof)
    if usable.sum() < 50 or len(np.unique(y_train[usable])) < 2:
        log.warning("not enough out-of-fold predictions to calibrate; scores stay uncalibrated")
        return None
    estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        oof[usable], y_train[usable]
    )
    log.info("calibrated on %d out-of-fold predictions over %d folds", int(usable.sum()), folds)
    return isotonic_from_sklearn(estimator)


def _fold_assignments(y: np.ndarray, folds: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified fold ids, so every fold carries a comparable bad rate."""
    assignments = np.empty(len(y), dtype=int)
    for value in np.unique(y):
        idx = np.nonzero(y == value)[0]
        rng.shuffle(idx)
        assignments[idx] = np.arange(len(idx)) % folds
    return assignments


def _calibration_report(
    datasets: Mapping[str, pd.DataFrame],
    label: str,
    best: Candidate,
    calibrator: IsotonicCurve | None,
) -> pd.DataFrame:
    """Predicted versus realised bad rate per score decile."""
    rows = []
    for name in DATASETS:
        if name not in datasets:
            continue
        y = datasets[name][label].to_numpy(dtype=float)
        raw = best.scores[name]
        if len(y) == 0:
            continue
        edges = M.band_edges(raw, 10)
        band = np.clip(np.searchsorted(edges, raw, side="left"), 0, len(edges))
        calibrated = calibrator.predict(raw) if calibrator is not None else raw
        for value in sorted(set(band.tolist())):
            mask = band == value
            rows.append(
                {
                    "dataset": name,
                    "band": int(value),
                    "rows": int(mask.sum()),
                    "raw_mean": round(float(raw[mask].mean()), 6),
                    "calibrated_mean": round(float(calibrated[mask].mean()), 6),
                    "actual_bad_rate": round(float(y[mask].mean()), 6),
                }
            )
    report = pd.DataFrame(rows)
    if len(report):
        report["raw_error"] = (report["raw_mean"] - report["actual_bad_rate"]).abs().round(6)
        report["calibrated_error"] = (report["calibrated_mean"] - report["actual_bad_rate"]).abs().round(6)
    return report


def _diagnostics(metrics: pd.DataFrame, best_name: str) -> dict:
    rows = metrics[metrics["model"] == best_name].set_index("dataset")
    out: dict[str, float] = {}
    for metric in ("ks", "auc"):
        if {"train", "test"}.issubset(rows.index):
            out[f"train_test_{metric}_gap"] = round(float(rows.loc["train", metric] - rows.loc["test", metric]), 6)
        if {"test", "holdout"}.issubset(rows.index):
            out[f"test_holdout_{metric}_gap"] = round(float(rows.loc["test", metric] - rows.loc["holdout", metric]), 6)
    gap = out.get("train_test_ks_gap")
    if gap is not None:
        out["overfit_verdict"] = (
            "severe" if gap > 0.10 else "moderate" if gap > 0.05 else "acceptable"
        )
    return out
