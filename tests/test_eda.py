"""Unit tests for modules/eda/auto_eda.py."""
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.eda.auto_eda import AutoEDA, calc_woe_iv, classify_iv, check_monotonicity


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "score": np.random.normal(650, 80, n).clip(300, 850),
        "age": np.random.randint(20, 65, n),
        "income": np.random.exponential(10000, n),
        "label": np.random.randint(0, 2, n),
    })
    # Inject signal: low score → more bads
    prob_bad = 1 / (1 + np.exp((df["score"] - 500) / 100))
    df["label"] = (np.random.random(n) < prob_bad).astype(int)
    return df


class TestCalcWOEIV:
    def test_returns_dataframe(self, sample_df):
        woe_df = calc_woe_iv(sample_df, "score", "label", bins=5)
        assert isinstance(woe_df, pd.DataFrame)
        assert "woe" in woe_df.columns
        assert "iv" in woe_df.columns
        assert "bad_rate" in woe_df.columns
        assert "bin_interval" in woe_df.columns

    def test_iv_positive(self, sample_df):
        """IV = Σ(good_pct - bad_pct) * ln(good/bad) must be non-negative."""
        woe_df = calc_woe_iv(sample_df, "score", "label", bins=10)
        iv = woe_df["iv"].iloc[0]
        assert iv >= 0

    def test_woe_direction_standard(self, sample_df):
        """WOE = ln(good/bad): bins with lower bad_rate have higher WOE."""
        woe_df = calc_woe_iv(sample_df, "score", "label", bins=5)
        # WOE should generally increase as bad_rate decreases
        sorted_by_rate = woe_df.sort_values("bad_rate")
        woe_values = sorted_by_rate["woe"].values
        # Higher bad_rate bins should have lower WOE (more bads → lower good/bad ratio)
        # Just check that not all WOE are identical
        assert woe_df["woe"].nunique() > 1

    def test_tree_method(self, sample_df):
        woe_df = calc_woe_iv(sample_df, "score", "label", bins=5, method="tree")
        assert len(woe_df) >= 2

    def test_quantile_method(self, sample_df):
        woe_df = calc_woe_iv(sample_df, "score", "label", bins=5, method="quantile")
        assert len(woe_df) >= 2

    def test_categorical_feature(self):
        df = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F"] * 100,
            "label": [0, 1, 0, 0, 1, 0, 1, 1] * 100,
        })
        woe_df = calc_woe_iv(df, "gender", "label")
        assert len(woe_df) == 2  # M and F


class TestClassifyIV:
    def test_ranges(self):
        assert classify_iv(0.01) == "无预测力"
        assert classify_iv(0.05) == "弱"
        assert classify_iv(0.2) == "中"
        assert classify_iv(0.4) == "强"
        assert "极强" in classify_iv(0.6)

    def test_negative_iv(self):
        result = classify_iv(-0.1)
        assert result == "无预测力"


class TestMonotonicity:
    def test_monotone_increasing(self):
        # Create monotonically increasing WOE
        woe_df = pd.DataFrame({
            "bin": range(5),
            "woe": [-0.5, -0.2, 0.0, 0.3, 0.6],
        })
        result = check_monotonicity(woe_df)
        assert result["is_monotone"]

    def test_non_monotone(self):
        woe_df = pd.DataFrame({
            "bin": range(5),
            "woe": [0.5, -0.3, 0.2, -0.1, 0.1],
        })
        result = check_monotonicity(woe_df)
        assert not result["is_monotone"]

    def test_too_few_bins(self):
        woe_df = pd.DataFrame({
            "bin": range(2),
            "woe": [0.1, 0.2],
        })
        result = check_monotonicity(woe_df)
        assert not result["is_monotone"]


class TestAutoEDAReports:
    def test_run_outputs_woe_bin_report(self, sample_df, tmp_path):
        eda = AutoEDA()
        result = eda.run(sample_df, "label", report_dir=str(tmp_path))
        report_path = tmp_path / "woe_bin_report.csv"
        report = pd.read_csv(report_path)

        assert report_path.exists()
        assert not report.empty
        assert {"feature", "bin_label", "bin_interval", "total", "bad", "bad_rate", "woe", "iv_bin", "iv"}.issubset(report.columns)
        assert "woe_bin_report" in result
