from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from riskflow.data.schema import infer_schema
from riskflow.policy.decision import DecisionPolicy
from riskflow.policy.mining import mine_rules, mine_single_rules, rule_stats
from riskflow.policy.predicates import Predicate, Rule, evaluate_any, first_hit_labels
from riskflow.policy.thresholds import Cutoff, SegmentCutoff, evaluate_cutoff, search_global_cutoff
from riskflow.policy.validation import backtest, combined_effect, marginal_contribution, select_stable, stability
from riskflow.settings import CutoffSettings, RuleSettings

LABEL = "is_bad"


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, np.nan, 50.0],
            "grade": ["A", "B", None, "C", "A"],
            "odd name [x]": [1.0, 2.0, 3.0, 4.0, 5.0],
            LABEL: [0, 1, 0, 1, 1],
        }
    )


# ------------------------------------------------------------------ predicates


def test_missing_values_never_satisfy_a_comparison(frame):
    hit = Predicate("amount", "ge", 5.0).evaluate(frame)
    assert hit.tolist() == [True, True, True, False, True]
    assert Predicate("amount", "le", 1e9).evaluate(frame)[3] == False  # noqa: E712


def test_null_checks_are_explicit(frame):
    assert Predicate("amount", "is_null").evaluate(frame).tolist() == [False, False, False, True, False]
    assert Predicate("grade", "is_null").evaluate(frame).tolist() == [False, False, True, False, False]


def test_categorical_operators(frame):
    assert Predicate("grade", "eq", "A").evaluate(frame).tolist() == [True, False, False, False, True]
    assert Predicate("grade", "in", ("A", "B")).evaluate(frame).tolist() == [True, True, False, False, True]
    # a null is neither in nor not-in the set; it simply does not match
    assert Predicate("grade", "not_in", ("A",)).evaluate(frame).tolist() == [False, True, False, True, False]


def test_awkward_column_names_survive_serialisation(frame):
    """The whole reason predicates are structured rather than parsed strings."""
    rule = Rule((Predicate("odd name [x]", "gt", 3.0),))
    restored = Rule.from_dict(json.loads(json.dumps(rule.to_dict())))
    np.testing.assert_array_equal(restored.evaluate(frame), rule.evaluate(frame))
    assert restored == rule


def test_rules_are_conjunctions(frame):
    rule = Rule((Predicate("amount", "ge", 20.0), Predicate("grade", "eq", "A")))
    assert rule.evaluate(frame).tolist() == [False, False, False, False, True]
    assert rule.features == ("amount", "grade")


def test_unknown_operator_is_rejected():
    with pytest.raises(ValueError, match="unknown operator"):
        Predicate("amount", "approximately", 1.0)
    with pytest.raises(ValueError, match="at least one predicate"):
        Rule(())


def test_referencing_a_missing_column_names_it(frame):
    with pytest.raises(KeyError, match="ghost"):
        Predicate("ghost", "ge", 1.0).evaluate(frame)


def test_first_hit_reports_one_reason_per_row(frame):
    rules = [
        Rule((Predicate("amount", "ge", 50.0),), rule_id="big"),
        Rule((Predicate("grade", "eq", "A"),), rule_id="grade-a"),
    ]
    labels = first_hit_labels(rules, frame)
    assert labels[4] == rules[0].describe(), "the earlier rule wins when both fire"
    assert labels[0] == rules[1].describe()
    assert labels[1] == ""
    assert evaluate_any(rules, frame).tolist() == [True, False, False, False, True]


# ---------------------------------------------------------------------- mining


def test_mining_finds_a_planted_pocket(applications, settings):
    schema = infer_schema(applications, LABEL, time_col="apply_time", id_col="application_id")
    features = [f for f in schema.features if not f.startswith("noise")]
    rules = mine_rules(applications, schema, features, settings.rules)

    assert rules, "the generator plants a high-risk pocket that mining should find"
    for rule in rules:
        stats = rule_stats(rule, applications, LABEL)
        assert stats["coverage"] <= settings.rules.max_coverage
        assert stats["lift"] >= settings.rules.min_lift
        assert stats["hits"] >= settings.rules.min_hits


def test_threshold_grid_reaches_below_the_coverage_cap(applications):
    """A grid whose finest step equals max_coverage can never produce a narrow rule."""
    schema = infer_schema(applications, LABEL, time_col="apply_time", id_col="application_id")
    settings = RuleSettings(max_coverage=0.05)
    rules = mine_single_rules(applications, schema, ["debt_ratio"], settings)
    coverages = [rule_stats(r, applications, LABEL)["coverage"] for r in rules]
    assert min(coverages) < settings.max_coverage / 2


def test_overlapping_rules_are_collapsed(applications, settings):
    schema = infer_schema(applications, LABEL, time_col="apply_time", id_col="application_id")
    rules = mine_rules(applications, schema, list(schema.features), settings.rules)
    masks = [rule.evaluate(applications) for rule in rules]
    for i, left in enumerate(masks):
        for right in masks[i + 1 :]:
            smaller = min(left.sum(), right.sum())
            if smaller:
                assert (left & right).sum() / smaller < settings.rules.max_overlap


# ------------------------------------------------------------------ validation


def test_a_rule_that_only_works_on_train_is_rejected():
    rng = np.random.default_rng(21)
    train = pd.DataFrame({"x": rng.random(1000), LABEL: 0})
    # a pocket that exists only in train
    train.loc[train["x"] > 0.98, LABEL] = 1
    train.loc[rng.random(1000) < 0.1, LABEL] = 1
    other = pd.DataFrame({"x": rng.random(600), LABEL: (rng.random(600) < 0.1).astype(int)})

    rule = Rule((Predicate("x", "gt", 0.98),), rule_id="train-only")
    datasets = {"train": train, "test": other, "holdout": other}
    table = stability(backtest([rule], datasets, LABEL), RuleSettings())

    assert not bool(table["stable"].iloc[0])
    assert table["verdict"].iloc[0] != "stable"
    assert select_stable([rule], table) == []


def test_a_genuinely_stable_rule_survives():
    rng = np.random.default_rng(22)
    datasets = {}
    for name, size in (("train", 4000), ("test", 1200), ("holdout", 1200)):
        x = rng.random(size)
        y = (rng.random(size) < np.where(x > 0.985, 0.9, 0.08)).astype(int)
        datasets[name] = pd.DataFrame({"x": x, LABEL: y})

    rule = Rule((Predicate("x", "gt", 0.985),), rule_id="real")
    table = stability(backtest([rule], datasets, LABEL), RuleSettings(min_hits=5))
    assert bool(table["stable"].iloc[0]), table["verdict"].iloc[0]


def test_marginal_contribution_does_not_double_count(applications):
    rules = [
        Rule((Predicate("max_delinquency", "ge", 3.0),), rule_id="a"),
        Rule((Predicate("max_delinquency", "ge", 2.0),), rule_id="b"),
    ]
    table = marginal_contribution(rules, applications, LABEL)
    assert table["new_hits"].sum() == int(rules[1].evaluate(applications).sum())
    assert table["new_hits"].iloc[1] < table["hits"].iloc[1]


def test_combined_effect_lowers_the_approved_bad_rate(applications):
    rule = Rule((Predicate("max_delinquency", "ge", 4.0),), rule_id="severe")
    table = combined_effect([rule], {"train": applications}, LABEL)
    row = table.iloc[0]
    assert row["approved_bad_rate"] < row["base_bad_rate"]
    assert row["bad_rate_reduction"] > 0


# --------------------------------------------------------------------- cutoffs


@pytest.fixture
def scored_datasets():
    rng = np.random.default_rng(23)
    datasets, scores = {}, {}
    for name, size in (("train", 4000), ("test", 1500), ("holdout", 1500)):
        risk = rng.random(size)
        y = (rng.random(size) < risk * 0.35).astype(int)
        datasets[name] = pd.DataFrame({LABEL: y, "channel": rng.choice(["app", "agent"], size)})
        scores[name] = risk
    return datasets, scores


def test_cutoff_bands_are_ordered_and_exhaustive(scored_datasets):
    datasets, scores = scored_datasets
    search = search_global_cutoff(datasets, scores, LABEL, CutoffSettings())
    performance = search.performance[search.performance["dataset"] == "test"].set_index("action")

    assert performance["share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert performance.loc["reject", "bad_rate"] > performance.loc["review", "bad_rate"]
    assert performance.loc["review", "bad_rate"] > performance.loc["approve", "bad_rate"]
    assert search.chosen.review_at <= search.chosen.reject_at


def test_cutoff_thresholds_come_from_the_training_distribution(scored_datasets):
    datasets, scores = scored_datasets
    search = search_global_cutoff(datasets, scores, LABEL, CutoffSettings(reject_rate_grid=(0.05,), review_rate_grid=(0.10,)))
    assert search.chosen.reject_at == pytest.approx(float(np.quantile(scores["train"], 0.95)))
    assert search.chosen.review_at == pytest.approx(float(np.quantile(scores["train"], 0.85)))


def test_a_score_with_no_signal_falls_back_to_the_narrowest_band(scored_datasets, caplog):
    datasets, _ = scored_datasets
    rng = np.random.default_rng(24)
    noise = {name: rng.random(len(frame)) for name, frame in datasets.items()}
    search = search_global_cutoff(datasets, noise, LABEL, CutoffSettings())
    rejected = search.performance[(search.performance["dataset"] == "test") & (search.performance["action"] == "reject")]
    assert rejected["share"].iloc[0] == pytest.approx(0.03, abs=0.01)


# -------------------------------------------------------------------- decision


def test_rules_outrank_the_score(frame):
    policy = DecisionPolicy(
        global_cutoff=Cutoff(reject_at=0.9, review_at=0.5),
        reject_rules=(Rule((Predicate("grade", "eq", "A"),), rule_id="grade-a"),),
    )
    decisions = policy.decide(frame, [0.1, 0.1, 0.6, 0.95, 0.1])

    assert decisions["decision"].tolist() == ["reject", "approve", "review", "reject", "reject"]
    assert decisions["rejected_by_rule"].tolist() == [True, False, False, False, True]
    assert decisions["reason"].iloc[0].startswith("rule:")
    assert decisions["reason"].iloc[3] == "score at or above the reject threshold"


def test_segment_overrides_apply_only_to_their_segment(frame):
    policy = DecisionPolicy(
        global_cutoff=Cutoff(reject_at=0.9, review_at=0.5),
        segment_cutoffs=(SegmentCutoff("grade", "A", Cutoff(reject_at=0.2, review_at=0.1)),),
    )
    decisions = policy.decide(frame, [0.3, 0.3, 0.3, 0.3, 0.3])

    assert decisions["decision"].tolist() == ["reject", "approve", "approve", "approve", "reject"]
    assert decisions["segment"].iloc[0] == "grade=A"
    assert decisions["segment"].iloc[1] == ""


def test_a_segment_override_can_target_missing_values(frame):
    policy = DecisionPolicy(
        global_cutoff=Cutoff(reject_at=0.9, review_at=0.8),
        segment_cutoffs=(SegmentCutoff("grade", None, Cutoff(reject_at=0.1, review_at=0.05)),),
    )
    decisions = policy.decide(frame, [0.5] * 5)
    assert decisions["decision"].tolist() == ["approve", "approve", "reject", "approve", "approve"]


def test_policy_round_trips_through_json(frame):
    policy = DecisionPolicy(
        global_cutoff=Cutoff(reject_at=0.8, review_at=0.4),
        reject_rules=(Rule((Predicate("amount", "ge", 30.0), Predicate("grade", "in", ("A", "C"))),),),
        segment_cutoffs=(SegmentCutoff("grade", "B", Cutoff(reject_at=0.6, review_at=0.3)),),
    )
    restored = DecisionPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    scores = [0.1, 0.5, 0.7, 0.9, 0.95]
    pd.testing.assert_frame_equal(restored.decide(frame, scores), policy.decide(frame, scores))


def test_score_length_mismatch_is_caught(frame):
    policy = DecisionPolicy(global_cutoff=Cutoff(reject_at=0.9, review_at=0.5))
    with pytest.raises(ValueError, match="3 scores for 5 rows"):
        policy.decide(frame, [0.1, 0.2, 0.3])
