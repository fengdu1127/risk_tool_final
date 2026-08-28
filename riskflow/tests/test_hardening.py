"""Regression tests for defects found by adversarial review.

Each test names the failure it prevents. They are grouped by what the defect
would have cost in production, because that is what decides whether a future
change may relax one.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from riskflow.features.binning import BinningSettings, BinningError, canonical, fit_categorical, fit_numeric
from riskflow.policy.decision import DecisionPolicy
from riskflow.policy.predicates import Predicate, Rule
from riskflow.policy.thresholds import Cutoff, SegmentCutoff
from riskflow.policy.validation import backtest
from riskflow.registry import Registry
from riskflow.settings import Settings

LABEL = "is_bad"


@pytest.fixture
def minimal_bundle():
    """The smallest bundle that is still a real, loadable artefact."""
    from riskflow.bundle import ScoringBundle
    from riskflow.data.schema import DatasetSchema
    from riskflow.features.space import woe_space
    from riskflow.features.woe import WoeTransformer
    from riskflow.models.predictors import LinearScorer
    from riskflow.monitoring.drift import DriftBaseline

    rng = np.random.default_rng(2)
    values = rng.normal(size=400)
    y = (rng.random(400) < 0.3).astype(int)
    woe = WoeTransformer(binnings={"x": fit_numeric(values, y, BinningSettings(), "x")})
    return ScoringBundle(
        schema=DatasetSchema(label=LABEL, numeric=("x",), categorical=()),
        woe=woe,
        space=woe_space(("x",)),
        predictor=LinearScorer(features=("x",), coefficients=(0.5,), intercept=0.0),
        policy=DecisionPolicy(global_cutoff=Cutoff(reject_at=0.9, review_at=0.5)),
        drift=DriftBaseline.fit(pd.DataFrame({"x": values}), woe, ["x"]),
    )



# ---------------------------------------------------------------------------
# Silent wrong answers: the policy decides, but on the wrong basis
# ---------------------------------------------------------------------------


def test_a_non_finite_score_is_refused_rather_than_approved():
    """Failing open is the one outcome a credit decision must never have.

    Every comparison against a NaN is false, so an unusable score used to slide
    through all three bands and land on 'approve'.
    """
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    policy = DecisionPolicy(global_cutoff=Cutoff(reject_at=0.9, review_at=0.5))

    with pytest.raises(ValueError, match="not finite"):
        policy.decide(frame, [np.nan, 0.95, 0.1])
    with pytest.raises(ValueError, match="not finite"):
        policy.decide(frame, [np.inf, 0.95, 0.1])

    # the healthy case is untouched
    assert policy.decide(frame, [0.1, 0.6, 0.95])["decision"].tolist() == ["approve", "review", "reject"]


def test_a_missing_policy_column_is_refused_rather_than_ignored():
    """An absent segment column used to silently revert everyone to the global cutoff."""
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    policy = DecisionPolicy(
        global_cutoff=Cutoff(reject_at=0.9, review_at=0.5),
        segment_cutoffs=(SegmentCutoff("channel", "app", Cutoff(reject_at=0.2, review_at=0.1)),),
    )
    assert policy.required_columns() == ("channel",)
    with pytest.raises(KeyError, match="channel"):
        policy.decide(frame, [0.5, 0.5])


def test_required_columns_covers_rules_and_segments_without_duplicates():
    policy = DecisionPolicy(
        global_cutoff=Cutoff(reject_at=0.9, review_at=0.5),
        reject_rules=(Rule((Predicate("a", "ge", 1.0), Predicate("b", "is_null"))),),
        segment_cutoffs=(SegmentCutoff("a", "x", Cutoff(0.2, 0.1)),),
    )
    assert policy.required_columns() == ("a", "b")


# ---------------------------------------------------------------------------
# Train/serve skew: the same applicant, encoded differently at scoring time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(12, "12"), (12.0, "12"), (np.int64(12), "12"), (np.float64(12.0), "12"),
     (12.5, "12.5"), (True, "True"), ("app", "app"), (None, None), (np.nan, None)],
)
def test_canonical_keys_survive_an_int_to_float_round_trip(value, expected):
    """A column of ints comes back as float64 the moment one row goes missing."""
    assert canonical(value) == expected


def test_a_numeric_segment_still_matches_after_pandas_widens_the_dtype():
    """The exact skew this defends against: `term=12` vs a batch holding 12.0."""
    train = pd.DataFrame({"term": [12, 24, 12, 36]})
    batch = pd.DataFrame({"term": [12.0, 24.0, 12.0, 36.0]})
    segment = SegmentCutoff("term", "12", Cutoff(reject_at=0.1, review_at=0.05))

    assert segment.matches(train).sum() == 2
    assert segment.matches(batch).sum() == 2


def test_a_numeric_category_keeps_its_woe_after_the_dtype_widens():
    rng = np.random.default_rng(0)
    levels = pd.Series([1, 2, 3] * 300)
    y = (rng.random(900) < 0.3).astype(int)
    binning = fit_categorical(levels, y, BinningSettings(missing_min_rows=50), "tier")

    as_int = binning.transform(pd.Series([1, 2, 3]))
    as_float = binning.transform(pd.Series([1.0, 2.0, 3.0]))
    as_text = binning.transform(pd.Series(["1", "2", "3"]))

    np.testing.assert_allclose(as_int, as_float)
    np.testing.assert_allclose(as_int, as_text)
    assert binning.other_woe not in set(as_float), "no level should fall through to the pooled bucket"


# ---------------------------------------------------------------------------
# Confidently wrong numbers
# ---------------------------------------------------------------------------


def test_a_single_class_label_cannot_produce_a_binning():
    """It used to return a binning with an enormous, entirely fabricated IV."""
    rng = np.random.default_rng(1)
    with pytest.raises(BinningError, match="single class"):
        fit_numeric(rng.normal(size=500), np.zeros(500), BinningSettings(), "x")
    with pytest.raises(BinningError, match="single class"):
        fit_categorical(rng.choice(["a", "b"], 500), np.ones(500), BinningSettings(), "g")


def test_duplicate_rule_ids_are_refused_before_validation():
    """A collision let one rule's evidence wave two rules through the gates."""
    frame = pd.DataFrame({"a": np.arange(100.0), LABEL: (np.arange(100) > 90).astype(int)})
    rules = [Rule((Predicate("a", "gt", 1.0),), rule_id="X"), Rule((Predicate("a", "gt", 50.0),), rule_id="X")]
    with pytest.raises(ValueError, match="rule ids must be unique"):
        backtest(rules, {"train": frame}, LABEL)


# ---------------------------------------------------------------------------
# Artefact portability and hygiene
# ---------------------------------------------------------------------------


def test_metadata_nan_becomes_null_but_a_model_weight_nan_is_refused(tmp_path, minimal_bundle):
    """Python emits bare NaN; no other JSON parser accepts it.

    Descriptive metadata may legitimately hold a metric that could not be
    computed, so it is nulled. A non-finite model weight is a broken model and
    must not reach disk.
    """
    from dataclasses import replace

    from riskflow.models.predictors import LinearScorer

    bundle = replace(minimal_bundle, metadata={"metrics": [{"auc": float("nan"), "ks": 0.4}]})
    path = bundle.save(tmp_path / "bundle.json")
    text = path.read_text(encoding="utf-8")

    assert "NaN" not in text and "Infinity" not in text
    # a strict parser must accept it, which is the whole point of the format
    json.loads(text, parse_constant=lambda token: pytest.fail(f"invalid JSON token {token!r}"))
    assert json.loads(text)["metadata"]["metrics"][0]["auc"] is None

    broken = replace(bundle, predictor=LinearScorer(features=("x",), coefficients=(float("nan"),), intercept=0.0))
    with pytest.raises(ValueError, match="non-finite"):
        broken.save(tmp_path / "broken.json")


def test_a_run_name_cannot_escape_the_registry(tmp_path):
    registry = Registry(tmp_path / "runs")
    for name in ("../escaped", "nested/run", "..", "PRODUCTION"):
        with pytest.raises(ValueError, match="invalid run name"):
            registry.new_run(name)
    assert registry.new_run("legitimate").path.parent == registry.root


# ---------------------------------------------------------------------------
# Configuration mistakes surface at the point they are made
# ---------------------------------------------------------------------------


def test_a_fractional_value_for_a_whole_number_setting_is_caught_immediately():
    """It used to be accepted, then blow up much later inside range()."""
    with pytest.raises(TypeError, match="whole number"):
        Settings().merged({"model": {"search_iterations": 3.7}})

    # JSON has no integer type, so 4.0 is a legitimate spelling of 4
    merged = Settings().merged({"model": {"search_iterations": 4.0}})
    assert merged.model.search_iterations == 4
    assert isinstance(merged.model.search_iterations, int)


def test_a_categorical_rule_keeps_firing_after_the_dtype_widens():
    """The same skew as the segment case, but on a hard reject rule.

    A rule mined against an integer column used to match nothing once pandas
    read that column back as float — and a rule that matches nothing raises
    nothing, so a hard decline would simply stop happening.
    """
    mined = pd.DataFrame({"tier": [1, 2, 3] * 100})
    widened = pd.DataFrame({"tier": [1.0, 2.0, 3.0] * 100})
    rule = Rule((Predicate("tier", "eq", 3),))

    assert rule.evaluate(mined).sum() == 100
    assert rule.evaluate(widened).sum() == 100
    assert rule.evaluate(pd.DataFrame({"tier": ["3"] * 300})).sum() == 300


def test_membership_predicates_are_canonicalised_too():
    frame = pd.DataFrame({"tier": [1.0, 2.0, 3.0, np.nan]})
    inside = Predicate("tier", "in", (1, 2))
    outside = Predicate("tier", "not_in", (1, 2))

    assert inside.evaluate(frame).tolist() == [True, True, False, False]
    # a null is neither inside nor outside the set
    assert outside.evaluate(frame).tolist() == [False, False, True, False]


def test_predicate_canonicalisation_survives_serialisation():
    original = Predicate("tier", "eq", 3)
    restored = Predicate.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.value == "3"


def test_mined_categorical_rules_use_canonical_levels():
    from riskflow.data.schema import infer_schema
    from riskflow.policy.mining import mine_single_rules
    from riskflow.settings import RuleSettings

    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"tier": rng.choice([1, 2, 3], 600).astype(object)})
    frame[LABEL] = (rng.random(600) < 0.3).astype(int)
    schema = infer_schema(frame, LABEL)

    rules = mine_single_rules(frame, schema, ["tier"], RuleSettings())
    levels = {p.value for r in rules for p in r.predicates if p.op == "eq"}
    assert levels == {"1", "2", "3"}, "levels must be stored in canonical form"

    widened = frame.copy()
    widened["tier"] = widened["tier"].astype(float)
    for rule in rules:
        assert rule.evaluate(frame).sum() == rule.evaluate(widened).sum()


def test_the_production_pointer_is_replaced_atomically(tmp_path, minimal_bundle):
    """A crash mid-write must not leave production unresolvable.

    Scoring reads this pointer to find out which run is live, so a truncated
    write would take the whole scoring path down until someone re-promoted.
    """
    from unittest.mock import patch

    registry = Registry(tmp_path / "runs")
    for name in ("first", "second"):
        run = registry.new_run(name)
        minimal_bundle.save(run.bundle_path)
        run.summary_path.write_text("{}", encoding="utf-8")

    registry.promote("first")
    assert registry.production().name == "first"

    # a failure during the swap must leave the previous pointer intact
    with patch("riskflow.registry.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            registry.promote("second")
    assert registry.production().name == "first"
    assert not list(registry.root.glob(".PRODUCTION.*")), "a staging file was left behind"

    registry.promote("second")
    assert registry.production().name == "second"


def test_an_empty_pointer_is_reported_not_silently_ignored(tmp_path):
    registry = Registry(tmp_path / "runs")
    registry.root.mkdir(parents=True)
    registry.pointer_path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        registry.production()
