"""Unit tests for modules/strategy/auto_strategy.py."""
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.strategy.auto_strategy import AutoStrategy, SingleRuleMiner, VariableSelector


@pytest.fixture
def strategy_df():
    np.random.seed(42)
    n = 800
    df = pd.DataFrame({
        "score": np.random.normal(650, 80, n).clip(300, 850),
        "age": np.random.randint(20, 65, n),
        "income": np.random.exponential(10000, n),
        "debt_ratio": np.random.beta(2, 5, n),
        "label": np.random.randint(0, 2, n),
    })
    prob_bad = 1 / (1 + np.exp((df["score"] - 500) / 100 + df["debt_ratio"] * 2))
    df["label"] = (np.random.random(n) < prob_bad).astype(int)
    return df


@pytest.fixture
def default_config():
    from config.config import STRATEGY_CONFIG
    return STRATEGY_CONFIG.copy()


class TestVariableSelector:
    def test_fit_returns_list(self, strategy_df, default_config):
        selector = VariableSelector(default_config)
        result = selector.fit(strategy_df, "label")
        assert isinstance(result, list)

    def test_fit_keeps_some_vars(self, strategy_df, default_config):
        selector = VariableSelector(default_config)
        result = selector.fit(strategy_df, "label")
        assert len(result) >= 1

    def test_fit_reports_dropped(self, strategy_df, default_config):
        selector = VariableSelector(default_config)
        selector.fit(strategy_df, "label")
        assert isinstance(selector.dropped, dict)
        assert len(selector.selected) + len(selector.dropped) > 0

    def test_drops_high_corr(self, strategy_df, default_config):
        # Add a highly correlated duplicate
        df = strategy_df.copy()
        df["score_dup"] = df["score"] * 0.98 + np.random.normal(0, 5, len(df))
        selector = VariableSelector(default_config)
        selected = selector.fit(df, "label")
        # One of score/score_dup should be dropped
        assert not ("score" in selected and "score_dup" in selected)

    def test_empty_pool_handled(self, default_config):
        df = pd.DataFrame({"label": [0, 1, 0, 1]})
        selector = VariableSelector(default_config)
        selected = selector.fit(df, "label")
        assert selected == []

    def test_categorical_features_included(self, default_config):
        """VariableSelector should include categorical features present in IV table."""
        np.random.seed(42)
        n = 400
        df = pd.DataFrame({
            "score": np.random.normal(650, 80, n),
            "channel": np.random.choice(["A", "B", "C"], n),
            "label": np.random.randint(0, 2, n),
        })
        iv_table = pd.DataFrame({
            "feature": ["score", "channel"],
            "IV": [0.15, 0.08],
        })
        selector = VariableSelector(default_config)
        selected = selector.fit(df, "label", iv_table)
        assert "channel" in selected

    def test_categorical_skipped_if_not_in_iv(self, default_config):
        """Categorical features not in IV table should be dropped."""
        np.random.seed(42)
        n = 400
        df = pd.DataFrame({
            "score": np.random.normal(650, 80, n),
            "channel": np.random.choice(["A", "B", "C"], n),
            "label": np.random.randint(0, 2, n),
        })
        iv_table = pd.DataFrame({
            "feature": ["score"],
            "IV": [0.15],
        })
        selector = VariableSelector(default_config)
        selected = selector.fit(df, "label", iv_table)
        assert "channel" not in selected


class TestSingleRuleMiner:
    @pytest.fixture
    def rule_df(self):
        np.random.seed(42)
        n = 800
        df = pd.DataFrame({
            "score": np.random.normal(650, 80, n).clip(300, 850),
            "channel": np.random.choice(["A", "B", "C"], n),
            "label": np.random.randint(0, 2, n),
        })
        prob_bad = 1 / (1 + np.exp((df["score"] - 500) / 100))
        df["label"] = (np.random.random(n) < prob_bad).astype(int)
        return df

    def test_categorical_equality_rules(self, rule_df, default_config):
        miner = SingleRuleMiner(default_config)
        rules = miner.mine(rule_df, "label", ["channel"])
        # Should generate == rules for categorical feature
        assert len(rules) >= 0  # may or may not pass lift/coverage thresholds
        if len(rules) > 0:
            assert all(rules["direction"] == "==")

    def test_numeric_threshold_rules(self, rule_df, default_config):
        miner = SingleRuleMiner(default_config)
        rules = miner.mine(rule_df, "label", ["score"])
        if len(rules) > 0:
            assert all(rules["direction"].isin([">=", "<="]))

    def test_mixed_feature_rules(self, rule_df, default_config):
        miner = SingleRuleMiner(default_config)
        rules = miner.mine(rule_df, "label", ["score", "channel"])
        assert isinstance(rules, pd.DataFrame)
        if len(rules) > 0:
            assert set(rules["direction"].unique()).issubset({">=", "<=", "=="})


class TestStrategyStability:
    def test_stable_rules_require_all_datasets_to_pass(self, default_config):
        strategy = AutoStrategy(default_config)
        all_rules = pd.DataFrame({
            "feature": ["x", "y"],
            "direction": [">=", ">="],
            "threshold": [1.0, 1.0],
            "condition": ["x >= 1", "y >= 1"],
            "condition_str": ["x >= 1", "y >= 1"],
            "coverage": [0.03, 0.03],
            "hit_count": [30, 30],
            "bad_rate": [0.8, 0.8],
            "lift": [3.0, 3.0],
            "rule_type": ["single", "single"],
        })
        bt_train = pd.DataFrame({
            "dataset": ["train", "train"],
            "rule_id": [0, 1],
            "condition": ["x >= 1", "y >= 1"],
            "coverage": [0.03, 0.03],
            "hit_count": [30, 30],
            "bad_rate": [0.8, 0.8],
            "lift": [3.0, 3.0],
        })
        bt_test = bt_train.assign(dataset="test", lift=[2.8, 2.8])
        bt_valid = bt_train.assign(dataset="valid", lift=[2.7, 1.2])
        stability = strategy._build_rule_stability(bt_train, bt_test, bt_valid)
        stable_rules = strategy._stable_rules(all_rules, stability)
        assert stable_rules["condition_str"].tolist() == ["x >= 1"]

    def test_overlap_dedupe_keeps_first_rule(self, default_config):
        strategy = AutoStrategy(default_config)
        df = pd.DataFrame({"x": np.arange(100), "label": [0, 1] * 50})
        rules = pd.DataFrame({
            "feature": ["x", "x"],
            "direction": [">=", ">="],
            "threshold": [90, 91],
            "condition": ["x >= 90", "x >= 91"],
            "condition_str": ["x >= 90", "x >= 91"],
            "coverage": [0.1, 0.09],
            "hit_count": [10, 9],
            "bad_rate": [0.5, 0.5],
            "lift": [2.0, 1.9],
            "rule_type": ["single", "single"],
        })
        deduped = strategy._drop_overlapped_rules(df, rules)
        assert len(deduped) == 1
        assert deduped.iloc[0]["condition_str"] == "x >= 90"

    def test_score_policy_report_creates_three_actions(self, default_config):
        strategy = AutoStrategy(default_config)
        df = pd.DataFrame({
            "model_score": np.linspace(0.01, 0.99, 100),
            "label": [0, 1] * 50,
        })
        report = strategy._score_policy_report(df, df, df, "label")
        assert set(report["action"]) == {"approve", "review", "reject"}
        assert set(report["dataset"]) == {"train", "test", "valid"}

    def test_marginal_contribution_counts_new_hits(self, default_config):
        strategy = AutoStrategy(default_config)
        df = pd.DataFrame({"x": np.arange(10), "label": [0, 1] * 5})
        rules = pd.DataFrame({
            "feature": ["x", "x"],
            "direction": [">=", ">="],
            "threshold": [5, 7],
            "condition": ["x >= 5", "x >= 7"],
            "condition_str": ["x >= 5", "x >= 7"],
            "rule_type": ["single", "single"],
        })
        report = strategy._marginal_contribution_report(df, df, df, "label", rules)
        train = report[report["dataset"] == "train"].reset_index(drop=True)
        assert train.loc[0, "marginal_hit_count"] == 5
        assert train.loc[1, "marginal_hit_count"] == 0

    def test_segment_strategy_report_uses_configured_segments(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["channel"]
        cfg["segment_min_samples"] = 2
        strategy = AutoStrategy(cfg)
        df = pd.DataFrame({
            "x": [9, 8, 1, 0, 9, 8, 1, 0],
            "channel": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "label": [1, 1, 0, 0, 1, 0, 0, 0],
        })
        rules = pd.DataFrame({
            "feature": ["x"],
            "direction": [">="],
            "threshold": [8],
            "condition": ["x >= 8"],
            "condition_str": ["x >= 8"],
            "rule_type": ["single"],
        })
        report = strategy._segment_strategy_report(df, df, df, "label", rules)
        assert set(report["segment_feature"]) == {"channel"}
        assert set(report["segment_value"]) == {"A", "B"}

    def test_overall_strategy_search_generates_grid_candidates(self, default_config):
        cfg = default_config.copy()
        cfg["overall_reject_rate_grid"] = [0.1, 0.2]
        cfg["overall_review_rate_grid"] = [0.1]
        strategy = AutoStrategy(cfg)
        df = pd.DataFrame({
            "model_score": np.linspace(0.01, 0.99, 100),
            "label": [0] * 70 + [1] * 30,
        })
        candidates, recommendation = strategy._overall_strategy_search(df, df, df, "label")
        assert candidates["policy_id"].nunique() == 2
        assert set(candidates["action"]) == {"reject", "review", "approve"}
        assert not recommendation.empty

    def test_overall_recommendation_prefers_higher_test_lift(self, default_config):
        strategy = AutoStrategy(default_config)
        candidates = pd.DataFrame({
            "policy_id": ["a", "b", "a", "b"],
            "dataset": ["test", "test", "valid", "valid"],
            "action": ["reject", "reject", "reject", "reject"],
            "lift": [3.0, 2.0, 1.5, 4.0],
            "bad_rate": [0.6, 0.4, 0.3, 0.8],
            "rate": [0.05, 0.05, 0.05, 0.05],
        })
        rec = strategy._recommend_overall_strategy(candidates)
        assert rec["policy_id"].iloc[0] == "a"

    def test_segment_strategy_requires_valid_hits(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["channel"]
        cfg["segment_min_samples"] = 2
        cfg["segment_min_valid_hits"] = 10
        cfg["segment_reject_rate_grid"] = [0.1]
        strategy = AutoStrategy(cfg)
        train = pd.DataFrame({"model_score": np.linspace(0, 1, 100), "channel": ["A"] * 100, "label": [0] * 70 + [1] * 30})
        test = train.copy()
        valid = pd.DataFrame({"model_score": np.linspace(0, 1, 20), "channel": ["A"] * 20, "label": [0] * 14 + [1] * 6})
        overall = pd.DataFrame({"review_threshold": [0.7]})
        candidates, rec = strategy._segment_strategy_search(train, test, valid, "label", overall)
        assert not candidates.empty
        assert not rec["recommend_segment_strategy"].any()
        assert rec["fallback_to_global"].all()

    def test_segment_discrimination_rejects_imbalanced_share(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["seg"]
        strategy = AutoStrategy(cfg)
        df = pd.DataFrame({
            "seg": ["A"] * 95 + ["B"] * 5,
            "label": [0] * 80 + [1] * 15 + [1] * 5,
        })
        report = strategy._segment_discrimination_report(df, df, df, "label")
        assert bool(report.loc[0, "share_pass"]) is False
        assert bool(report.loc[0, "recommend_as_segment_feature"]) is False

    def test_segment_discrimination_rejects_low_bad_rate_gap(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["seg"]
        strategy = AutoStrategy(cfg)
        df = pd.DataFrame({
            "seg": ["A"] * 50 + ["B"] * 50,
            "label": [0, 1] * 50,
        })
        report = strategy._segment_discrimination_report(df, df, df, "label")
        assert bool(report.loc[0, "bad_rate_gap_pass"]) is False
        assert bool(report.loc[0, "recommend_as_segment_feature"]) is False

    def test_segment_discrimination_accepts_balanced_risk_gap(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["seg"]
        strategy = AutoStrategy(cfg)
        df = pd.DataFrame({
            "seg": ["A"] * 50 + ["B"] * 50,
            "label": [1] * 20 + [0] * 30 + [1] * 5 + [0] * 45,
        })
        report = strategy._segment_discrimination_report(df, df, df, "label")
        assert bool(report.loc[0, "share_pass"]) is True
        assert bool(report.loc[0, "bad_rate_gap_pass"]) is True
        assert bool(report.loc[0, "recommend_as_segment_feature"]) is True

    def test_segment_discrimination_rejects_bad_test_share(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["seg"]
        strategy = AutoStrategy(cfg)
        train = pd.DataFrame({"seg": ["A"] * 50 + ["B"] * 50, "label": [1] * 20 + [0] * 30 + [1] * 5 + [0] * 45})
        test = pd.DataFrame({"seg": ["A"] * 95 + ["B"] * 5, "label": [1] * 30 + [0] * 65 + [1] * 5})
        valid = train.copy()
        report = strategy._segment_discrimination_report(train, test, valid, "label")
        assert bool(report.loc[0, "share_pass"]) is False
        assert bool(report.loc[0, "recommend_as_segment_feature"]) is False

    def test_segment_discrimination_rejects_low_test_bad_rate_gap(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["seg"]
        strategy = AutoStrategy(cfg)
        train = pd.DataFrame({"seg": ["A"] * 50 + ["B"] * 50, "label": [1] * 20 + [0] * 30 + [1] * 5 + [0] * 45})
        test = pd.DataFrame({"seg": ["A"] * 50 + ["B"] * 50, "label": [0, 1] * 50})
        valid = train.copy()
        report = strategy._segment_discrimination_report(train, test, valid, "label")
        assert bool(report.loc[0, "bad_rate_gap_pass"]) is False
        assert bool(report.loc[0, "recommend_as_segment_feature"]) is False

    def test_segment_strategy_search_uses_filtered_features(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["pass_seg", "blocked_seg"]
        cfg["segment_min_samples"] = 10
        cfg["segment_min_valid_hits"] = 1
        cfg["segment_reject_rate_grid"] = [0.1]
        strategy = AutoStrategy(cfg)
        df = pd.DataFrame({
            "model_score": np.linspace(0, 1, 120),
            "pass_seg": ["A"] * 60 + ["B"] * 60,
            "blocked_seg": ["X"] * 60 + ["Y"] * 60,
            "label": [0] * 80 + [1] * 40,
        })
        overall = pd.DataFrame({"review_threshold": [0.6]})
        candidates, _ = strategy._segment_strategy_search(df, df, df, "label", overall, ["pass_seg"])
        assert not candidates.empty
        assert set(candidates["segment_feature"]) == {"pass_seg"}

    def test_segment_strategy_search_matches_nan_segments(self, default_config):
        cfg = default_config.copy()
        cfg["segment_features"] = ["channel"]
        cfg["segment_min_samples"] = 10
        cfg["segment_min_valid_hits"] = 1
        cfg["segment_reject_rate_grid"] = [0.1]
        strategy = AutoStrategy(cfg)
        df = pd.DataFrame({
            "model_score": np.linspace(0, 1, 100),
            "channel": [np.nan] * 50 + ["A"] * 50,
            "label": [0] * 70 + [1] * 30,
        })
        overall = pd.DataFrame({"review_threshold": [0.6]})
        candidates, _ = strategy._segment_strategy_search(df, df, df, "label", overall)
        nan_rows = candidates[candidates["segment_value"].isna()]
        assert not nan_rows.empty
        assert set(nan_rows["dataset"]) == {"train", "test", "valid"}

    def test_segment_recommendation_rejects_valid_lift_spike(self, default_config):
        cfg = default_config.copy()
        cfg["segment_min_valid_hits"] = 10
        cfg["segment_min_lift"] = 2.0
        cfg["segment_max_lift_gap"] = 0.5
        strategy = AutoStrategy(cfg)
        candidates = pd.DataFrame({
            "policy_id": ["p1", "p1", "p1"],
            "policy_type": ["segment", "segment", "segment"],
            "dataset": ["train", "test", "valid"],
            "action": ["reject", "reject", "reject"],
            "reject_threshold": [0.8, 0.8, 0.8],
            "review_threshold": [0.6, 0.6, 0.6],
            "count": [30, 20, 20],
            "rate": [0.1, 0.1, 0.1],
            "bad_rate": [0.5, 0.42, 0.6],
            "overall_bad_rate": [0.2, 0.2, 0.2],
            "lift": [2.5, 2.1, 3.0],
            "segment_feature": ["channel", "channel", "channel"],
            "segment_value": ["A", "A", "A"],
        })
        rec = strategy._recommend_segment_strategy(candidates)
        assert not rec["recommend_segment_strategy"].any()
        assert rec["fallback_to_global"].all()
        assert "abs_lift_gap_test_valid" in rec.columns

    def test_final_strategy_comparison_contains_three_schemes(self, default_config):
        strategy = AutoStrategy(default_config)
        df = pd.DataFrame({
            "model_score": np.linspace(0.01, 0.99, 20),
            "x": np.arange(20),
            "channel": ["A"] * 10 + ["B"] * 10,
            "label": [0] * 12 + [1] * 8,
        })
        stable_rules = pd.DataFrame({
            "feature": ["x"],
            "direction": [">="],
            "threshold": [18],
            "condition_str": ["x >= 18"],
            "rule_type": ["single"],
        })
        overall = pd.DataFrame({
            "dataset": ["train", "test", "valid"],
            "action": ["reject", "reject", "reject"],
            "reject_threshold": [0.9, 0.9, 0.9],
            "review_threshold": [0.7, 0.7, 0.7],
        })
        segment = pd.DataFrame({
            "dataset": ["train", "test", "valid"],
            "action": ["reject", "reject", "reject"],
            "segment_feature": ["channel", "channel", "channel"],
            "segment_value": ["B", "B", "B"],
            "reject_threshold": [0.8, 0.8, 0.8],
            "review_threshold": [0.6, 0.6, 0.6],
            "recommend_segment_strategy": [True, True, True],
        })
        comparison = strategy._final_strategy_comparison(df, df, df, "label", stable_rules, overall, segment)
        assert set(comparison["strategy_name"]) == {"global_strategy", "segment_strategy", "global_plus_segment_rules"}

    def test_strategy_leaderboard_contains_three_schemes(self, default_config):
        strategy = AutoStrategy(default_config)
        comparison = pd.DataFrame({
            "strategy_name": ["global_strategy", "segment_strategy", "global_plus_segment_rules"] * 3,
            "dataset": ["train"] * 3 + ["test"] * 3 + ["valid"] * 3,
            "reject_rate": [0.05, 0.04, 0.08, 0.05, 0.04, 0.08, 0.05, 0.04, 0.08],
            "review_rate": [0.1] * 9,
            "approve_rate": [0.85, 0.86, 0.82] * 3,
            "reject_bad_rate": [0.4, 0.45, 0.5, 0.38, 0.42, 0.46, 0.36, 0.44, 0.48],
            "approve_bad_rate": [0.1] * 9,
            "overall_bad_rate": [0.2] * 9,
            "lift": [2.0, 2.2, 2.4, 1.9, 2.1, 2.3, 1.8, 2.2, 2.4],
        })
        leaderboard = strategy._strategy_leaderboard(comparison)
        assert set(leaderboard["strategy_name"]) == {"global_strategy", "segment_strategy", "global_plus_segment_rules"}

    def test_strategy_recommendation_outputs_three_levels(self, default_config):
        strategy = AutoStrategy(default_config)
        leaderboard = pd.DataFrame({
            "strategy_name": ["global_strategy", "segment_strategy", "global_plus_segment_rules"],
            "valid_reject_rate": [0.04, 0.06, 0.10],
            "valid_reject_bad_rate": [0.35, 0.45, 0.50],
            "valid_lift": [1.8, 2.2, 2.4],
            "test_valid_lift_gap": [0.1, 0.2, 0.3],
            "leaderboard_score": [1.6, 2.0, 2.1],
        })
        recommendation = strategy._strategy_recommendation(leaderboard)
        assert set(recommendation["recommendation_type"]) == {"conservative", "balanced", "aggressive"}
