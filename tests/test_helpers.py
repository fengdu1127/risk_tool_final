"""Unit tests for utils/helpers.py core functions."""
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.helpers import (
    calc_ks,
    calc_auc,
    calc_psi,
    calc_lift,
    model_report,
    score_bin_report,
    split_dataset_with_recent_valid,
)


class TestKS:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = calc_ks(y_true, y_prob)
        assert ks == 1.0

    def test_random_prediction(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.uniform(0, 1, 1000)
        ks = calc_ks(y_true, y_prob)
        assert 0 <= ks <= 0.3  # random should give low KS

    def test_all_zeros_label(self):
        y_true = np.zeros(100)
        y_prob = np.random.uniform(0, 1, 100)
        ks = calc_ks(y_true, y_prob)
        assert np.isnan(ks) or ks == 0.0


class TestAUC:
    def test_perfect(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        auc = calc_auc(y_true, y_prob)
        assert auc == 1.0

    def test_random(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.uniform(0, 1, 1000)
        auc = calc_auc(y_true, y_prob)
        assert 0.3 < auc < 0.7


class TestPSI:
    def test_identical_distributions(self):
        x = np.random.normal(0, 1, 1000)
        psi = calc_psi(x, x)
        assert psi < 0.01

    def test_different_distributions(self):
        x1 = np.random.normal(0, 1, 1000)
        x2 = np.random.normal(3, 1, 1000)
        psi = calc_psi(x1, x2)
        assert psi > 0.5  # very different

    def test_symmetric(self):
        x1 = np.random.normal(0, 1, 1000)
        x2 = np.random.normal(0, 1, 1000)
        psi_ab = calc_psi(x1, x2)
        psi_ba = calc_psi(x2, x1)
        assert abs(psi_ab - psi_ba) < 0.1


class TestLift:
    def test_lift_greater_than_one(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        mask = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # all bads
        lift = calc_lift(y_true, mask)
        assert lift > 1.5

    def test_lift_equal_one(self):
        y_true = np.array([0, 1, 0, 1])
        mask = np.array([1, 1, 0, 0])
        lift = calc_lift(y_true, mask)
        assert lift == 1.0  # masked bad rate = 1/2 = 0.5 = overall

    def test_lift_empty_mask(self):
        y_true = np.array([0, 0, 1, 1])
        mask = np.zeros(4)
        lift = calc_lift(y_true, mask)
        assert lift == 0.0


class TestModelReport:
    def test_structure(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        report = model_report(y_true, y_prob, "test")
        assert report["dataset"] == "test"
        assert "AUC" in report
        assert "KS" in report
        assert "Gini" in report
        assert 0 <= report["AUC"] <= 1
        assert 0 <= report["KS"] <= 1


class TestRecentValidSplit:
    def test_latest_three_months_go_to_valid(self):
        dates = pd.date_range("2025-01-01", "2025-12-31", periods=365)
        df = pd.DataFrame({
            "apply_time": dates,
            "feature": np.arange(len(dates)),
            "label": [0, 1] * 182 + [0],
        })
        train, test, valid, info = split_dataset_with_recent_valid(
            df,
            "label",
            "apply_time",
            valid_months=3,
            min_valid_samples=10,
        )
        assert valid["apply_time"].min() > info["valid_start"]
        assert max(train["apply_time"].max(), test["apply_time"].max()) <= info["valid_start"]
        assert valid["apply_time"].min() > train["apply_time"].max()

    def test_invalid_time_col_raises(self):
        df = pd.DataFrame({"apply_time": ["bad-date", "2025-01-02"], "x": [1, 2], "label": [0, 1]})
        with pytest.raises(ValueError, match="unparseable"):
            split_dataset_with_recent_valid(df, "label", "apply_time", min_valid_samples=1)

    def test_history_split_is_stratified(self):
        dates = pd.date_range("2025-01-01", "2025-12-31", periods=500)
        y = np.array([0] * 400 + [1] * 100)
        df = pd.DataFrame({"apply_time": dates, "x": np.arange(500), "label": y})
        train, test, _, _ = split_dataset_with_recent_valid(
            df,
            "label",
            "apply_time",
            valid_months=2,
            min_valid_samples=10,
        )
        assert abs(train["label"].mean() - test["label"].mean()) < 0.05


class TestScoreBins:
    def test_score_bin_report_fields(self):
        y = np.array([0, 1] * 50)
        prob = np.linspace(0.01, 0.99, 100)
        report = score_bin_report(y, prob, "valid", n_bins=5)
        expected_cols = {
            "dataset",
            "score_bin",
            "bin_order",
            "score_bin_interval",
            "count",
            "bad_count",
            "raw_bad_rate",
            "bad_rate",
            "monotone_bad_rate",
            "bad_rate_monotone",
            "raw_bad_rate_monotone",
            "lift",
            "monotone_lift",
            "cum_bad_capture",
            "score_min",
            "score_max",
        }
        assert expected_cols.issubset(report.columns)
        assert report["dataset"].eq("valid").all()
        assert report["cum_bad_capture"].between(0, 1).all()
        assert report["score_bin_interval"].notna().all()

    def test_score_bin_report_uses_shared_breakpoints(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9])
        breakpoints = np.array([-np.inf, 0.5, np.inf])
        report = score_bin_report(y, prob, "test", breakpoints=breakpoints)
        assert set(report["score_bin_interval"]) == {"(-inf, 0.500000]", "(0.500000, inf]"}

    def test_score_bin_report_adds_monotone_bad_rate(self):
        y = np.array([1] * 8 + [0] * 2 + [1] + [0] * 9 + [1] * 3 + [0] * 7)
        prob = np.linspace(0.99, 0.01, len(y))
        report = score_bin_report(y, prob, "valid", n_bins=3)
        assert not report["raw_bad_rate"].is_monotonic_decreasing
        assert report["monotone_bad_rate"].is_monotonic_decreasing
        assert report["bad_rate_monotone"].all()
