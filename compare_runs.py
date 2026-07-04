"""Compare two pipeline runs: model metrics, overfit gaps, strategy outcomes.

Usage:
    python compare_runs.py reports/run_a reports/run_b [--output diff.csv]
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import get_logger

logger = get_logger("COMPARE")


def _load_summary(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"run_summary.json not found in {run_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(run_dir: str, *parts) -> pd.DataFrame:
    path = os.path.join(run_dir, *parts)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _collect_metrics(run_dir: str) -> dict:
    """Flatten one run into {metric_name: value}."""
    summary = _load_summary(run_dir)
    metrics = {
        "n_total": summary.get("n_total"),
        "best_model": summary.get("best_model"),
        "n_selected_vars": summary.get("n_selected_vars"),
        "n_stable_rules": summary.get("n_stable_rules"),
    }
    for row in summary.get("model_metrics", []):
        prefix = f"{row.get('model')}_{row.get('dataset')}"
        metrics[f"{prefix}_KS"] = row.get("KS")
        metrics[f"{prefix}_AUC"] = row.get("AUC")
    for key, value in (summary.get("overfit_diagnostics") or {}).items():
        metrics[key] = value

    leaderboard = _load_csv(run_dir, "strategy", "strategy_leaderboard.csv")
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        metrics["best_strategy"] = best.get("strategy_name")
        for col in ["valid_reject_rate", "valid_reject_bad_rate", "valid_lift", "leaderboard_score"]:
            metrics[f"strategy_{col}"] = best.get(col)

    policy_path = os.path.join(run_dir, "strategy", "policy.json")
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
        thresholds = policy.get("score_thresholds") or {}
        metrics["policy_reject_threshold"] = thresholds.get("reject")
        metrics["policy_review_threshold"] = thresholds.get("review")
        metrics["policy_n_reject_rules"] = len(policy.get("reject_rules", []))
        metrics["policy_n_segment_overrides"] = len(policy.get("segment_overrides", []))
    return metrics


def compare_runs(run_a: str, run_b: str, output_path: str = None) -> pd.DataFrame:
    metrics_a = _collect_metrics(run_a)
    metrics_b = _collect_metrics(run_b)
    keys = list(dict.fromkeys(list(metrics_a) + list(metrics_b)))
    rows = []
    for key in keys:
        va, vb = metrics_a.get(key), metrics_b.get(key)
        delta = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = round(vb - va, 6)
        rows.append({"metric": key, "run_a": va, "run_b": vb, "delta_b_minus_a": delta})
    diff = pd.DataFrame(rows)
    print(f"\nrun_a = {run_a}\nrun_b = {run_b}\n")
    print(diff.to_string(index=False))
    if output_path:
        diff.to_csv(output_path, index=False)
        logger.info(f"saved {output_path}")
    return diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two pipeline runs")
    parser.add_argument("run_a", help="Baseline run directory")
    parser.add_argument("run_b", help="Candidate run directory")
    parser.add_argument("--output", default=None, help="Optional CSV path for the diff table")
    args = parser.parse_args()
    compare_runs(args.run_a, args.run_b, args.output)
