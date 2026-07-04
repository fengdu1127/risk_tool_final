"""End-to-end entrypoint for risk modeling, validation, and rule mining."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config.config import SPLIT_CONFIG, apply_config_overrides, config_snapshot
from modules.eda.auto_eda import AutoEDA
from modules.model.auto_model import AutoModel
from modules.reporting.dashboard import generate_dashboard
from modules.strategy.auto_strategy import AutoStrategy
from utils.helpers import (
    dataset_profile,
    get_logger,
    save_json,
    split_dataset,
    split_dataset_with_recent_valid,
    timestamp,
    validate_inputs,
)

logger = get_logger("PIPELINE")


def run_pipeline(
    data_path: str,
    label_col: str,
    algo: str = "both",
    tune: bool = True,
    feature_cols: list = None,
    output_dir: str = None,
    time_col: str = None,
    valid_months: int = None,
):
    ts = timestamp()
    output_dir = output_dir or f"reports/run_{ts}"
    eda_dir = f"{output_dir}/eda"
    model_dir = f"{output_dir}/model"
    strategy_dir = f"{output_dir}/strategy"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("risk modeling pipeline started")
    logger.info(f"output_dir={output_dir}")
    logger.info("=" * 60)

    df = pd.read_csv(data_path)
    validate_inputs(df, label_col, feature_cols)
    logger.info(f"data shape={df.shape}, bad_rate={df[label_col].mean():.4f}")

    resolved_time_col = time_col or SPLIT_CONFIG.get("time_col")
    resolved_valid_months = valid_months or SPLIT_CONFIG.get("valid_months", 3)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]
    if resolved_time_col and resolved_time_col in feature_cols:
        feature_cols = [c for c in feature_cols if c != resolved_time_col]

    if resolved_time_col:
        train_df, test_df, valid_df, split_info = split_dataset_with_recent_valid(
            df,
            label_col,
            time_col=resolved_time_col,
            valid_months=resolved_valid_months,
            train_ratio=SPLIT_CONFIG["train_ratio"],
            test_ratio=SPLIT_CONFIG["test_ratio"],
            random_state=SPLIT_CONFIG["random_state"],
            min_valid_samples=SPLIT_CONFIG.get("min_valid_samples", 50),
        )
    else:
        train_df, test_df, valid_df = split_dataset(
            df,
            label_col,
            train_ratio=SPLIT_CONFIG["train_ratio"],
            test_ratio=SPLIT_CONFIG["test_ratio"],
            holdout_ratio=SPLIT_CONFIG["holdout_ratio"],
            random_state=SPLIT_CONFIG["random_state"],
        )
        split_info = {
            "time_col": None,
            "valid_months": None,
            "train": dataset_profile(train_df, label_col),
            "test": dataset_profile(test_df, label_col),
            "valid": dataset_profile(valid_df, label_col),
        }

    train_df.to_csv(f"{output_dir}/split_train.csv", index=False)
    test_df.to_csv(f"{output_dir}/split_test.csv", index=False)
    valid_df.to_csv(f"{output_dir}/split_valid.csv", index=False)
    pd.DataFrame([
        {"dataset": name, **split_info[name]}
        for name in ["train", "test", "valid"]
        if name in split_info
    ]).to_csv(f"{output_dir}/split_profile.csv", index=False)

    eda = AutoEDA()
    eda_results = eda.run(train_df, label_col, report_dir=eda_dir, time_col=resolved_time_col)
    iv_table = eda_results.get("iv_table", pd.DataFrame())

    model = AutoModel()
    model_results = model.run(
        train_df=train_df,
        test_df=test_df,
        valid_df=valid_df,
        label_col=label_col,
        feature_cols=feature_cols,
        algo=algo,
        tune=tune,
        report_dir=model_dir,
    )

    scored_datasets = model_results.get("scored_datasets", {})
    train_scored = scored_datasets.get("train", train_df)
    test_scored = scored_datasets.get("test", test_df)
    valid_scored = scored_datasets.get("valid", valid_df)

    strategy = AutoStrategy()
    strategy_results = strategy.run(
        train_df=train_scored,
        test_df=test_scored,
        valid_df=valid_scored,
        label_col=label_col,
        iv_table=iv_table,
        report_dir=strategy_dir,
        time_col=resolved_time_col,
    )

    summary = {
        "run_time": ts,
        "data_path": data_path,
        "label_col": label_col,
        "algo": algo,
        "tune": tune,
        "time_col": resolved_time_col,
        "valid_months": resolved_valid_months if resolved_time_col else None,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_valid": len(valid_df),
        "split_info": split_info,
        "best_model": model_results.get("best_algo", ""),
        "model_metrics": model_results.get("report_df", pd.DataFrame()).to_dict(orient="records"),
        "overfit_diagnostics": model_results.get("diagnostics", {}),
        "n_selected_vars": len(strategy_results.get("selected_vars", [])),
        "n_single_rules": len(strategy_results.get("single_rules", [])),
        "n_tree_rules": len(strategy_results.get("tree_rules", [])),
        "n_stable_rules": len(strategy_results.get("stable_rules", [])),
        "dashboard_path": f"{output_dir}/dashboard.html",
        "config": config_snapshot(),
    }
    save_json(summary, f"{output_dir}/run_summary.json")
    generate_dashboard(output_dir)
    logger.info(f"pipeline complete | output_dir={output_dir} | best_model={summary['best_model']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk modeling automation pipeline")
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--label", required=True, help="Binary label column")
    parser.add_argument("--algo", default="both", choices=["lr", "xgboost", "both"], help="Model algorithm")
    parser.add_argument("--features", nargs="+", default=None, help="Feature columns")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--time-col", default=None, help="Application time column for recent valid split")
    parser.add_argument("--valid-months", type=int, default=None, help="Latest N months used as valid set")
    parser.add_argument("--config", default=None, help="JSON file overriding config sections, e.g. {\"MODEL_CONFIG\": {\"optuna_trials\": 20}}")
    args = parser.parse_args()

    if args.config:
        applied = apply_config_overrides(args.config)
        logger.info(f"config overrides applied from {args.config}: {applied}")

    run_pipeline(
        data_path=args.data,
        label_col=args.label,
        algo=args.algo,
        tune=not args.no_tune,
        feature_cols=args.features,
        output_dir=args.output,
        time_col=args.time_col,
        valid_months=args.valid_months,
    )
