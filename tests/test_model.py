"""Unit tests for modules/model/auto_model.py."""
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.model.auto_model import WOEEncoder


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
