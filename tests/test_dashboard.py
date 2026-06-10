"""Unit tests for visual dashboard generation."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.reporting.dashboard import generate_dashboard


def test_generate_dashboard_creates_html(tmp_path):
    root = tmp_path / "run"
    eda_dir = root / "eda"
    model_dir = root / "model"
    strategy_dir = root / "strategy"
    eda_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    strategy_dir.mkdir(parents=True)

    (root / "run_summary.json").write_text(json.dumps({
        "n_total": 100,
        "best_model": "xgboost",
        "valid_months": 3,
    }), encoding="utf-8")
    pd.DataFrame({
        "dataset": ["train", "test", "valid"],
        "count": [70, 15, 15],
        "bad_rate": [0.2, 0.21, 0.22],
    }).to_csv(root / "split_profile.csv", index=False)
    pd.DataFrame({
        "model": ["xgboost", "xgboost", "xgboost"],
        "dataset": ["train", "test", "valid"],
        "AUC": [0.8, 0.75, 0.74],
        "KS": [0.5, 0.4, 0.39],
        "Gini": [0.6, 0.5, 0.48],
    }).to_csv(model_dir / "model_comparison.csv", index=False)
    pd.DataFrame({"train_test_ks_gap": [0.1], "test_valid_ks_gap": [0.01]}).to_csv(
        model_dir / "overfit_diagnostics.csv",
        index=False,
    )
    pd.DataFrame({
        "dataset": ["valid", "valid"],
        "score_bin": [1, 2],
        "bin_order": [1, 2],
        "score_bin_interval": ["(0.5, 1.0]", "(0.0, 0.5]"],
        "count": [10, 10],
        "bad_count": [4, 2],
        "raw_bad_rate": [0.4, 0.2],
        "bad_rate": [0.4, 0.2],
        "monotone_bad_rate": [0.4, 0.2],
        "lift": [2.0, 1.5],
        "monotone_lift": [2.0, 1.5],
        "cum_bad_capture": [0.6, 1.0],
        "bad_rate_gap_vs_train": [0.01, -0.02],
        "monotone_bad_rate_gap_vs_train": [0.01, -0.02],
        "bad_rate_monotone": [True, True],
        "raw_bad_rate_monotone": [True, True],
    }).to_csv(model_dir / "score_bins.csv", index=False)
    pd.DataFrame({
        "dataset": ["train", "test", "valid"],
        "n_bins": [10, 10, 10],
        "raw_bad_rate_monotone": [True, True, True],
        "monotone_bad_rate_monotone": [True, True, True],
        "max_abs_bad_rate_gap_vs_train": [0.0, 0.03, 0.04],
        "max_abs_monotone_bad_rate_gap_vs_train": [0.0, 0.02, 0.03],
    }).to_csv(model_dir / "score_bin_stability.csv", index=False)
    pd.DataFrame({
        "feature": ["score", "score"],
        "iv_rank": [1, 1],
        "bin_label": ["(-inf, 500]", "(500, inf]"],
        "bin_interval": ["(-inf, 500]", "(500, inf]"],
        "total": [50, 50],
        "bad": [20, 5],
        "bad_rate": [0.4, 0.1],
        "woe": [-0.5, 0.6],
        "iv_bin": [0.08, 0.06],
        "iv": [0.14, 0.14],
    }).to_csv(eda_dir / "woe_bin_report.csv", index=False)
    pd.DataFrame({
        "dataset": ["valid"],
        "feature": ["model_score"],
        "PSI": [0.02],
    }).to_csv(model_dir / "stability_report.csv", index=False)
    pd.DataFrame({
        "strategy_name": ["global_strategy"],
        "valid_reject_rate": [0.05],
        "valid_reject_bad_rate": [0.6],
        "valid_lift": [3.0],
        "test_valid_lift_gap": [0.1],
        "leaderboard_score": [3.4],
    }).to_csv(strategy_dir / "strategy_leaderboard.csv", index=False)
    pd.DataFrame({
        "recommendation_type": ["balanced"],
        "strategy_name": ["global_strategy"],
        "valid_reject_rate": [0.05],
        "valid_lift": [3.0],
        "leaderboard_score": [3.4],
    }).to_csv(strategy_dir / "strategy_recommendation.csv", index=False)
    pd.DataFrame({
        "strategy_name": ["global_strategy", "segment_strategy"],
        "dataset": ["valid", "valid"],
        "reject_rate": [0.05, 0.06],
        "review_rate": [0.10, 0.08],
        "approve_rate": [0.85, 0.86],
        "reject_bad_rate": [0.6, 0.62],
        "approve_bad_rate": [0.1, 0.09],
        "lift": [3.0, 3.1],
    }).to_csv(strategy_dir / "final_strategy_comparison.csv", index=False)
    pd.DataFrame({
        "policy_id": ["seg_channel_A_r0.030"],
        "dataset": ["valid"],
        "action": ["reject"],
        "segment_feature": ["channel"],
        "segment_value": ["A"],
        "reject_threshold": [0.8],
        "count": [12],
        "rate": [0.04],
        "bad_rate": [0.7],
        "lift": [3.5],
    }).to_csv(strategy_dir / "segment_strategy_candidates.csv", index=False)
    pd.DataFrame({
        "policy_id": ["seg_channel_A_r0.030"],
        "dataset": ["valid"],
        "action": ["reject"],
        "segment_feature": ["channel"],
        "segment_value": ["A"],
        "reject_threshold": [0.8],
        "count": [12],
        "rate": [0.04],
        "bad_rate": [0.7],
        "lift": [3.5],
        "recommend_segment_strategy": [True],
        "abs_lift_gap_test_valid": [0.2],
        "fallback_to_global": [False],
    }).to_csv(strategy_dir / "segment_strategy_recommendation.csv", index=False)
    pd.DataFrame({
        "segment_feature": ["channel"],
        "train_share_gap": [0.2],
        "valid_bad_rate_gap": [0.05],
        "stable_order": [True],
        "recommend_as_segment_feature": [True],
    }).to_csv(strategy_dir / "segment_discrimination.csv", index=False)
    pd.DataFrame({
        "rule_id": [0],
        "condition": ["x >= 1"],
        "lift_train": [2.0],
        "lift_test": [2.1],
        "lift_valid": [2.0],
        "stable_pass": [True],
    }).to_csv(strategy_dir / "rule_stability.csv", index=False)

    html_path = generate_dashboard(str(root))
    html_text = (root / "dashboard.html").read_text(encoding="utf-8")

    assert html_path.endswith("dashboard.html")
    assert "Risk Tool 可视化报告" in html_text
    assert "策略 Leaderboard" in html_text
    assert "客群区分度筛选" in html_text
    assert "单变量分箱风险表现" in html_text
    assert "模型评分切档效果" in html_text
    assert "评分十分箱单调稳定性" in html_text
    assert "最终策略明细" in html_text
    assert "分客群策略推荐明细" in html_text


def test_generate_dashboard_tolerates_missing_stability_columns(tmp_path):
    root = tmp_path / "run"
    model_dir = root / "model"
    strategy_dir = root / "strategy"
    model_dir.mkdir(parents=True)
    strategy_dir.mkdir(parents=True)

    (root / "run_summary.json").write_text(json.dumps({"n_total": 10}), encoding="utf-8")
    pd.DataFrame({"model": ["xgboost"], "dataset": ["valid"], "AUC": [0.7], "KS": [0.3], "Gini": [0.4]}).to_csv(
        model_dir / "model_comparison.csv",
        index=False,
    )
    pd.DataFrame({"train_test_ks_gap": [0.1], "test_valid_ks_gap": [0.02]}).to_csv(
        model_dir / "overfit_diagnostics.csv",
        index=False,
    )
    pd.DataFrame({"feature": ["model_score"]}).to_csv(model_dir / "stability_report.csv", index=False)

    html_path = generate_dashboard(str(root))

    assert html_path.endswith("dashboard.html")
    assert (root / "dashboard.html").exists()


def test_generate_dashboard_tolerates_empty_csv(tmp_path):
    root = tmp_path / "run"
    model_dir = root / "model"
    strategy_dir = root / "strategy"
    model_dir.mkdir(parents=True)
    strategy_dir.mkdir(parents=True)

    (root / "run_summary.json").write_text(json.dumps({"n_total": 10}), encoding="utf-8")
    (strategy_dir / "rule_stability.csv").write_text("", encoding="utf-8")

    html_path = generate_dashboard(str(root))

    assert html_path.endswith("dashboard.html")
    assert (root / "dashboard.html").exists()
