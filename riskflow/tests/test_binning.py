from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from riskflow.features.binning import (
    MISSING_BIN,
    OTHER_LEVEL,
    BinningError,
    binning_from_dict,
    fit_categorical,
    fit_numeric,
)
from riskflow.features.woe import WoeTransformer, monotonic_correlation
from riskflow.settings import BinningSettings

LABEL = "is_bad"


@pytest.fixture
def small_settings():
    return BinningSettings(max_bins=5, min_bin_rows=20, min_bin_fraction=0.02, missing_min_rows=20)


def test_woe_is_higher_where_risk_is_higher(small_settings):
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < x * 0.6).astype(int)
    binning = fit_numeric(x, y, small_settings, "x")

    rates = [s.bad_rate for s in binning.stats if s.index >= 0]
    woes = [s.woe for s in binning.stats if s.index >= 0]
    assert rates == sorted(rates), "bad rate should rise across bins for a monotone driver"
    assert woes == sorted(woes), "WOE must move with the bad rate, not against it"
    assert binning.transform([0.95])[0] > binning.transform([0.05])[0]


def test_monotone_merging_removes_violations():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 3000)
    # a genuine trend with one noisy pocket that breaks monotonicity
    risk = x * 0.5 + np.where((x > 0.4) & (x < 0.5), -0.25, 0.0)
    y = (rng.random(3000) < risk).astype(int)

    enforced = fit_numeric(x, y, BinningSettings(max_bins=8, min_bin_rows=30, enforce_monotonic=True), "x")
    rates = [s.bad_rate for s in enforced.stats if s.index >= 0]
    assert rates == sorted(rates) or rates == sorted(rates, reverse=True)
    assert abs(monotonic_correlation(enforced)) > 0.9


def test_missing_becomes_its_own_bin_when_common_enough(small_settings):
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(size=900), np.full(300, np.nan)])
    y = np.concatenate([(rng.random(900) < 0.1).astype(int), (rng.random(300) < 0.5).astype(int)])
    binning = fit_numeric(x, y, small_settings, "x")

    assert binning.missing_own_bin
    # missing rows are far riskier here, so their WOE must be well above the rest
    assert binning.missing_woe > max(binning.woe)
    assert binning.assign([np.nan])[0] == MISSING_BIN
    assert binning.transform([np.nan])[0] == binning.missing_woe


def test_rare_missing_falls_back_to_neutral():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(size=1000), np.full(5, np.nan)])
    y = (rng.random(1005) < 0.2).astype(int)
    binning = fit_numeric(x, y, BinningSettings(missing_min_rows=50), "x")

    assert not binning.missing_own_bin
    assert binning.transform([np.nan])[0] == 0.0


def test_bin_boundaries_are_right_closed(small_settings):
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 100, 1500)
    y = (rng.random(1500) < x / 200).astype(int)
    binning = fit_numeric(x, y, small_settings, "x")
    cut = binning.cuts[0]

    # a value exactly on a cut belongs to the lower bin
    assert binning.assign([cut])[0] == binning.assign([cut - 1e-9])[0]
    assert binning.assign([np.nextafter(cut, np.inf)])[0] == binning.assign([cut])[0] + 1


def test_categorical_pools_rare_levels_and_handles_unseen(small_settings):
    values = ["A"] * 500 + ["B"] * 400 + ["C"] * 3 + ["D"] * 2
    rng = np.random.default_rng(6)
    y = (rng.random(len(values)) < 0.2).astype(int)
    binning = fit_categorical(values, y, small_settings, "grade")

    assert OTHER_LEVEL in binning.levels, "rare levels should be pooled"
    assert set(binning.levels) == {"A", "B", OTHER_LEVEL}
    # a level never seen in training lands on the pooled WOE, not on a crash
    assert binning.transform(["Z"])[0] == binning.other_woe


def test_categorical_missing_is_distinct_from_unseen(small_settings):
    values = ["A"] * 400 + ["B"] * 400 + [None] * 200
    rng = np.random.default_rng(7)
    y = np.concatenate([
        (rng.random(800) < 0.1).astype(int),
        (rng.random(200) < 0.6).astype(int),
    ])
    binning = fit_categorical(values, y, small_settings, "grade")

    assert binning.transform([None])[0] == binning.missing_woe
    assert binning.transform([np.nan])[0] == binning.missing_woe
    assert binning.transform(["unseen"])[0] != binning.missing_woe


def test_unusable_columns_raise_binning_error(small_settings):
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    with pytest.raises(BinningError, match="constant"):
        fit_numeric(np.ones(100), y, small_settings, "flat")
    with pytest.raises(BinningError, match="every value is missing"):
        fit_numeric(np.full(100, np.nan), y, small_settings, "empty")
    with pytest.raises(BinningError, match="fewer than 2"):
        fit_categorical(["only"] * 100, y, small_settings, "single")


@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_binnings_round_trip_through_json(small_settings, kind):
    rng = np.random.default_rng(8)
    y = (rng.random(1000) < 0.3).astype(int)
    if kind == "numeric":
        values = np.where(rng.random(1000) < 0.1, np.nan, rng.normal(size=1000))
        original = fit_numeric(values, y, small_settings, "x")
        probe = pd.Series([np.nan, -2.0, 0.0, 2.0])
    else:
        values = rng.choice(["A", "B", "C"], 1000)
        original = fit_categorical(values, y, small_settings, "g")
        probe = pd.Series(["A", "B", "unseen", None])

    restored = binning_from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    np.testing.assert_array_equal(restored.transform(probe), original.transform(probe))


def test_transformer_skips_broken_columns_without_failing(toy_frame, small_settings):
    from riskflow.data.schema import infer_schema

    frame = toy_frame.copy()
    frame["constant"] = 1.0
    schema = infer_schema(frame, LABEL)
    transformer = WoeTransformer.fit(frame, schema, small_settings)

    assert "constant" in transformer.skipped
    assert "score_like" in transformer.features
    encoded = transformer.transform(frame, transformer.features)
    assert not encoded.isna().any().any(), "WOE encoding must leave no gaps"


def test_iv_ranks_a_real_driver_above_noise(toy_frame, small_settings):
    from riskflow.data.schema import infer_schema

    schema = infer_schema(toy_frame, LABEL)
    transformer = WoeTransformer.fit(toy_frame, schema, small_settings)
    iv = transformer.iv_table().set_index("feature")["iv"]
    assert iv["score_like"] > iv["sparse"]
