"""Unit tests for utils/binning.py."""
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.binning import tree_bin_thresholds, woe_stats


@pytest.fixture
def signal_data():
    np.random.seed(7)
    n = 1000
    x = pd.Series(np.random.normal(650, 80, n))
    prob_bad = 1 / (1 + np.exp((x - 600) / 80))
    y = pd.Series((np.random.random(n) < prob_bad).astype(int))
    return x, y


class TestTreeBinThresholds:
    def test_deterministic(self, signal_data):
        x, y = signal_data
        t1 = tree_bin_thresholds(x, y, bins=5)
        t2 = tree_bin_thresholds(x, y, bins=5)
        assert t1 == t2

    def test_threshold_count_bounded(self, signal_data):
        x, y = signal_data
        thresholds = tree_bin_thresholds(x, y, bins=5)
        assert 1 <= len(thresholds) <= 4  # max_leaf_nodes=5 -> at most 4 splits

    def test_thresholds_sorted(self, signal_data):
        x, y = signal_data
        thresholds = tree_bin_thresholds(x, y, bins=8)
        assert thresholds == sorted(thresholds)


class TestWOEStats:
    def test_woe_direction(self):
        # bin 0: all bad; bin 1: all good -> woe(0) < 0 < woe(1)
        bins = pd.Series([0] * 50 + [1] * 50)
        y = pd.Series([1] * 50 + [0] * 50)
        stats = woe_stats(bins, y)
        assert stats.loc[0, "woe"] < 0 < stats.loc[1, "woe"]

    def test_counts_and_bad_rate(self):
        bins = pd.Series([0, 0, 0, 1, 1, 1])
        y = pd.Series([1, 1, 0, 0, 0, 0])
        stats = woe_stats(bins, y)
        assert stats.loc[0, "total"] == 3
        assert stats.loc[0, "bad"] == 2
        assert abs(stats.loc[0, "bad_rate"] - 2 / 3) < 1e-9

    def test_null_bins_dropped(self):
        bins = pd.Series([0, 0, np.nan, 1, 1, np.nan])
        y = pd.Series([1, 0, 1, 0, 1, 0])
        stats = woe_stats(bins, y)
        assert stats["total"].sum() == 4

    def test_zero_count_bins_do_not_crash(self):
        # a bin with only good loans must not produce inf/NaN WOE
        bins = pd.Series([0] * 10 + [1] * 10)
        y = pd.Series([1] * 5 + [0] * 5 + [0] * 10)
        stats = woe_stats(bins, y)
        assert np.isfinite(stats["woe"]).all()
