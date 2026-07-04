"""Unit tests for modules/model/auto_model.py."""
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.model.auto_model import AutoModel, WOEEncoder, MISSING_BIN


@pytest.fixture
def woe_df():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "score": np.random.normal(650, 80, n).clip(300, 850),
        "age": np.random.randint(20, 65, n),
        "label": np.random.randint(0, 2, n),
    })
    prob_bad = 1 / (1 + np.exp((df["score"] - 500) / 100))
    df["label"] = (np.random.random(n) < prob_bad).astype(int)
    return df


class TestWOEEncoder:
    def test_fit_creates_mappings(self, woe_df):
        enc = WOEEncoder(bins=5)
        features = ["score", "age"]
        enc.fit(woe_df, features, "label")
        for feat in features:
            assert feat in enc.woe_maps
            assert feat in enc.thresholds
            assert len(enc.woe_maps[feat]) >= 2

    def test_transform_consistency(self, woe_df):
        """fit/transform must use same bin boundaries (tree thresholds)."""
        enc = WOEEncoder(bins=5)
        features = ["score"]
        enc.fit(woe_df, features, "label")

        # Transform on same data should use the stored tree thresholds
        transformed = enc.transform(woe_df, features)

        assert transformed is not None
        assert "score" in transformed.columns
        assert not transformed["score"].isna().any()

    def test_fit_transform_no_label_leak(self, woe_df):
        """Transform on different data should not fail."""
        enc = WOEEncoder(bins=5)
        features = ["score"]
        enc.fit(woe_df, features, "label")

        new_df = woe_df.drop(columns=["label"]).head(50)
        transformed = enc.transform(new_df, features)
        assert len(transformed) == 50
        assert "score" in transformed.columns

    def test_new_data_values_in_range(self, woe_df):
        """Values outside the fit range should still get a valid WOE."""
        enc = WOEEncoder(bins=5)
        features = ["score"]
        enc.fit(woe_df, features, "label")

        extreme_df = pd.DataFrame({
            "score": [-999, 0, 500, 1000, 99999],
        })
        transformed = enc.transform(extreme_df, features)
        assert not transformed["score"].isna().any()

    def test_empty_feature_list(self, woe_df):
        enc = WOEEncoder(bins=5)
        enc.fit(woe_df, [], "label")
        assert len(enc.woe_maps) == 0
        assert len(enc.thresholds) == 0

    def test_categorical_feature(self):
        """WOEEncoder should handle categorical features with direct WOE per category."""
        df = pd.DataFrame({
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F"] * 100,
            "city": ["BJ", "SH", "GZ", "BJ", "SH", "GZ", "BJ", "SH"] * 100,
            "label": [0, 1, 0, 0, 1, 0, 1, 1] * 100,
        })
        enc = WOEEncoder(bins=5)
        features = ["gender", "city"]
        enc.fit(df, features, "label")
        assert "gender" in enc.woe_maps
        assert "gender" in enc.cat_features
        assert len(enc.woe_maps["gender"]) == 2  # M, F

    def test_categorical_transform(self):
        """Categorical WOE transform should map values correctly."""
        df = pd.DataFrame({
            "gender": ["M", "F", "M", "F"] * 50,
            "label": [0, 1, 0, 0] * 50,
        })
        enc = WOEEncoder(bins=5)
        enc.fit(df, ["gender"], "label")
        transformed = enc.transform(df, ["gender"])
        assert "gender" in transformed.columns
        assert not transformed["gender"].isna().any()
        # Same category should get same WOE
        assert abs(transformed.loc[0, "gender"] - transformed.loc[2, "gender"]) < 1e-6

    def test_mixed_features(self, woe_df):
        """WOEEncoder should handle mixed numeric + categorical features."""
        df = woe_df.copy()
        df["gender"] = ["M", "F"] * (len(df) // 2)
        enc = WOEEncoder(bins=5)
        features = ["score", "age", "gender"]
        enc.fit(df, features, "label")
        assert "score" in enc.woe_maps  # numeric
        assert "gender" in enc.woe_maps  # categorical
        assert "gender" in enc.cat_features
        assert "score" not in enc.cat_features
        transformed = enc.transform(df, features)
        assert not transformed[["score", "age", "gender"]].isna().any(axis=None)


class TestMissingBin:
    @pytest.fixture
    def missing_df(self):
        """Numeric feature where missing rows are clearly riskier than the rest."""
        np.random.seed(42)
        n = 600
        df = pd.DataFrame({"x": np.random.normal(650, 80, n)})
        prob_bad = 1 / (1 + np.exp((df["x"] - 550) / 80))
        df["label"] = (np.random.random(n) < prob_bad).astype(int)
        missing_idx = df.index[:150]
        df.loc[missing_idx, "x"] = np.nan
        df.loc[missing_idx, "label"] = (np.random.random(150) < 0.8).astype(int)
        return df

    def test_missing_gets_own_bin(self, missing_df):
        enc = WOEEncoder(bins=5, missing_min_samples=50)
        enc.fit(missing_df, ["x"], "label")
        assert MISSING_BIN in enc.woe_maps["x"]
        # missing rows are risky -> WOE = ln(good/bad) should be negative
        assert enc.woe_maps["x"][MISSING_BIN] < 0

    def test_transform_maps_missing_to_missing_woe(self, missing_df):
        enc = WOEEncoder(bins=5, missing_min_samples=50)
        enc.fit(missing_df, ["x"], "label")
        new = pd.DataFrame({"x": [650.0, np.nan]})
        out = enc.transform(new, ["x"])
        assert abs(out.loc[1, "x"] - enc.woe_maps["x"][MISSING_BIN]) < 1e-9
        assert out.loc[0, "x"] != out.loc[1, "x"]

    def test_few_missing_falls_back_to_neutral(self):
        np.random.seed(0)
        n = 400
        df = pd.DataFrame({"x": np.random.normal(0, 1, n)})
        df["label"] = (np.random.random(n) < 0.3).astype(int)
        df.loc[df.index[:5], "x"] = np.nan  # below missing_min_samples
        enc = WOEEncoder(bins=5, missing_min_samples=50)
        enc.fit(df, ["x"], "label")
        assert MISSING_BIN not in enc.woe_maps["x"]
        out = enc.transform(pd.DataFrame({"x": [np.nan]}), ["x"])
        assert out.loc[0, "x"] == 0  # neutral WOE


class TestMonotoneConstraints:
    def test_directions_from_risk_correlation(self):
        np.random.seed(42)
        n = 800
        X = pd.DataFrame({
            "risk_up": np.arange(n, dtype=float),      # higher -> more bad
            "risk_down": -np.arange(n, dtype=float),   # higher -> less bad
            "noise": np.random.normal(0, 1, n),
            "channel": np.random.normal(0, 1, n),      # stands in for a WOE-encoded categorical
        })
        y = pd.Series((np.arange(n) + np.random.normal(0, 100, n) > n / 2).astype(int))
        # threshold above chance-level correlation (~1/sqrt(n)) so noise stays at 0
        model = AutoModel(config={"xgb_monotone": True, "xgb_monotone_min_abs_corr": 0.1})
        model.woe_encoder = WOEEncoder()
        model.woe_encoder.cat_features = {"channel"}
        constraints = model._xgb_monotone_constraints(X, y)
        assert constraints[0] == 1    # risk_up
        assert constraints[1] == -1   # risk_down
        assert constraints[2] == 0    # noise stays unconstrained
        assert constraints[3] == -1   # WOE-encoded categorical always -1

    def test_disabled_via_config(self):
        model = AutoModel(config={"xgb_monotone": False})
        assert model._xgb_monotone_constraints(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([0, 1])) is None
