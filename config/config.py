# Global configuration
import copy
import json

SPLIT_CONFIG = {
    "train_ratio": 0.70,
    "test_ratio": 0.15,
    "holdout_ratio": 0.15,
    "random_state": 42,
    "time_col": None,
    "valid_months": 3,
    "min_valid_samples": 50,
}

EDA_CONFIG = {
    "missing_threshold": 0.5,
    "iv_min": 0.02,
    "psi_stable": 0.1,
    "psi_warning": 0.25,
    "corr_threshold": 0.7,
    "vif_threshold": 10,
    "monotone_spearman_min": 0.6,
    "woe_bins": 10,
}

MODEL_CONFIG = {
    "algorithms": ["lr", "xgboost"],
    "cv_folds": 5,
    "optuna_trials": 50,
    "random_state": 42,
    "primary_metric": "KS",
    # missing values get their own WOE bin when train has at least this many
    "woe_missing_min_samples": 50,
    # constrain XGB predictions to be monotone in each feature's risk direction
    "xgb_monotone": True,
    "xgb_monotone_min_abs_corr": 0.02,
    "lr_default_params": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
    },
    "xgb_default_params": {
        "n_estimators": 80,
        "max_depth": 2,
        "learning_rate": 0.035,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 12,
        "gamma": 2.0,
        "reg_alpha": 0.3,
        "reg_lambda": 4.0,
        "eval_metric": "auc",
        "random_state": 42,
    },
    "scorecard": {
        "pdo": 20,
        "base_score": 600,
        "base_odds": 1 / 15,
    },
}

STRATEGY_CONFIG = {
    "corr_threshold": 0.7,
    "vif_threshold": 10,
    "monotone_spearman_min": 0.6,
    "psi_max": 0.1,
    "rule_max_coverage": 0.05,
    "rule_min_lift": 2.0,
    "rule_min_hit_count": 5,
    "rule_max_lift_drop": 0.35,
    "rule_overlap_threshold": 0.8,
    "score_reject_quantile": 0.95,
    "score_review_quantile": 0.85,
    "segment_features": ["channel", "city_tier", "loan_term"],
    "segment_min_samples": 50,
    "overall_reject_rate_grid": [0.03, 0.05, 0.08, 0.10],
    "overall_review_rate_grid": [0.05, 0.10, 0.15],
    "segment_reject_rate_grid": [0.03, 0.05, 0.08],
    "segment_min_valid_hits": 10,
    "segment_min_lift": 2.0,
    "segment_max_lift_gap": 0.5,
    "segment_min_share": 0.10,
    "segment_max_share": 0.60,
    "segment_max_share_gap": 0.45,
    "segment_min_bad_rate_gap": 0.03,
    "segment_require_stable_order": False,
    "segment_use_discrimination_filter": True,
    "tree_max_depth": 3,
    "tree_max_features": 3,
    "tree_min_samples_leaf": 50,
}

REPORT_CONFIG = {
    "output_dir": "reports",
    "fig_dpi": 150,
    "fig_format": "png",
    # False: scored_*.csv only keeps label/score/bin columns instead of full data copies
    "save_scored_full": False,
}

_ALL_CONFIGS = {
    "SPLIT_CONFIG": SPLIT_CONFIG,
    "EDA_CONFIG": EDA_CONFIG,
    "MODEL_CONFIG": MODEL_CONFIG,
    "STRATEGY_CONFIG": STRATEGY_CONFIG,
    "REPORT_CONFIG": REPORT_CONFIG,
}


def apply_config_overrides(path: str) -> dict:
    """Merge a JSON override file ({"MODEL_CONFIG": {"optuna_trials": 20}, ...})
    into the global config dicts in place. Returns the applied overrides."""
    # utf-8-sig tolerates the BOM that Windows editors often prepend
    with open(path, "r", encoding="utf-8-sig") as f:
        overrides = json.load(f)
    for section, values in overrides.items():
        if section not in _ALL_CONFIGS:
            raise KeyError(f"unknown config section '{section}', expected one of {list(_ALL_CONFIGS)}")
        if not isinstance(values, dict):
            raise ValueError(f"config section '{section}' must be an object")
        for key, value in values.items():
            if isinstance(_ALL_CONFIGS[section].get(key), dict) and isinstance(value, dict):
                _ALL_CONFIGS[section][key].update(value)
            else:
                _ALL_CONFIGS[section][key] = value
    return overrides


def config_snapshot() -> dict:
    """Deep copy of all effective config sections, for run reproducibility."""
    return copy.deepcopy(_ALL_CONFIGS)
