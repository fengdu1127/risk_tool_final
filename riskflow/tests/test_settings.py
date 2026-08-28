from __future__ import annotations

import json

import pytest

from riskflow.settings import Settings


def test_defaults_are_shared_but_not_shared_state():
    a, b = Settings(), Settings()
    assert a == b
    assert a.model.scorecard is not b.model.scorecard


def test_merge_is_functional_and_nested():
    base = Settings()
    merged = base.merged({"model": {"search_iterations": 3, "scorecard": {"pdo": 40}}})
    assert merged.model.search_iterations == 3
    assert merged.model.scorecard.pdo == 40
    # untouched branches keep their defaults, and the original is unchanged
    assert merged.split == base.split
    assert base.model.search_iterations == Settings().model.search_iterations


def test_unknown_keys_are_rejected_with_a_useful_message():
    with pytest.raises(KeyError, match="unknown setting 'sarch_iterations'"):
        Settings().merged({"model": {"sarch_iterations": 5}})
    with pytest.raises(KeyError, match="unknown setting 'modle'"):
        Settings().merged({"modle": {}})


def test_type_mistakes_are_caught_early():
    with pytest.raises(TypeError):
        Settings().merged({"report": {"make_plots": "yes"}})
    with pytest.raises(TypeError):
        Settings().merged({"cutoffs": {"reject_rate_grid": 0.05}})
    with pytest.raises(TypeError):
        Settings().merged({"model": "logistic"})


def test_tuple_settings_stay_tuples_after_override():
    merged = Settings().merged({"cutoffs": {"segment_features": ["channel", "city_tier"]}})
    assert merged.cutoffs.segment_features == ("channel", "city_tier")
    assert isinstance(merged.cutoffs.segment_features, tuple)


def test_round_trips_through_json(tmp_path):
    original = Settings().merged({"binning": {"max_bins": 5}, "rules": {"min_lift": 3.0}})
    path = tmp_path / "settings.json"
    original.to_json(path)
    assert Settings.load(path) == original


def test_load_tolerates_a_byte_order_mark(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"binning": {"max_bins": 4}}).encode())
    assert Settings.load(path).binning.max_bins == 4
