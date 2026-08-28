"""Full pipeline: train, persist, promote, score, monitor.

The central claim these tests defend is that the run directory is a complete,
self-sufficient deployment artefact — that a bundle loaded from disk, in a
process that never saw the training data, decides exactly what the training run
said it would.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from riskflow.bundle import ScoringBundle
from riskflow.data.synth import make_applications
from riskflow.monitoring.drift import alerts
from riskflow.registry import Registry
from riskflow.scoring import score_batch
from riskflow.train import train

LABEL = "is_bad"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from riskflow.settings import Settings

    settings = Settings().merged(
        {
            "split": {"time_col": "apply_time", "oot_months": 3},
            "model": {"search_iterations": 4, "cv_folds": 3},
            "cutoffs": {"segment_features": ["channel"], "segment_min_rows": 150},
            "report": {"make_plots": False},
        }
    )
    root = tmp_path_factory.mktemp("runs")
    data = make_applications(n_rows=6000, seed=31)
    result = train(
        data=data,
        label=LABEL,
        settings=settings,
        id_col="application_id",
        run_name="run_under_test",
        registry_root=root,
    )
    return result, root, data


def test_training_writes_a_complete_run(trained):
    result, _, _ = trained
    run = result.run

    complete, missing = run.is_complete()
    assert complete, f"run is missing {missing}"
    for table in ("screening", "binning", "model_metrics", "gains", "calibration", "cutoff_performance", "decisions"):
        assert run.table(table) is not None, f"expected table '{table}'"
    assert (run.path / "settings.json").exists()
    assert (run.path / "run.log").read_text(encoding="utf-8").strip()


def test_the_model_actually_discriminates(trained):
    result, _, _ = trained
    metrics = result.summary["model"]["metrics"]
    holdout = next(m for m in metrics if m["dataset"] == "holdout" and m["model"] == result.summary["model"]["algorithm"])
    assert holdout["auc"] > 0.65
    assert holdout["ks"] > 0.20


def test_screening_drops_the_planted_noise(trained):
    result, _, _ = trained
    selected = set(result.summary["features"]["selected"])
    assert "noise_score" not in selected
    assert "noise_flag" not in selected
    assert {"debt_ratio", "max_delinquency"} <= selected


def test_a_reloaded_bundle_scores_identically(trained):
    """No hidden state: disk round-trip must be exact, not merely close."""
    result, _, data = trained
    reloaded = ScoringBundle.load(result.run.bundle_path)

    before = result.bundle.score(data)
    after = reloaded.score(data)
    pd.testing.assert_frame_equal(before, after)


def test_the_bundle_is_pure_json_with_no_pickles(trained):
    result, _, _ = trained
    payload = json.loads(result.run.bundle_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert set(payload) >= {"schema", "woe", "space", "predictor", "policy", "drift"}
    # nothing in the run directory is a pickle
    assert not list(result.run.path.rglob("*.pkl"))
    assert not list(result.run.path.rglob("*.joblib"))


def test_scoring_needs_no_training_libraries(trained, monkeypatch):
    """Scoring must survive on a machine with neither sklearn nor xgboost."""
    import builtins

    result, _, data = trained
    expected = result.bundle.score(data.head(200))

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"sklearn", "xgboost", "scipy"}:
            raise ImportError(f"{name} is not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = ScoringBundle.load(result.run.bundle_path)
    pd.testing.assert_frame_equal(reloaded.score(data.head(200)), expected)


def test_decisions_are_ordered_by_risk(trained):
    result, _, data = trained
    scored = result.bundle.score(data)
    joined = scored.join(data[LABEL])
    rates = joined.groupby("decision")[LABEL].mean()

    assert rates["reject"] > rates["review"] > rates["approve"]
    assert (scored["credit_score"].corr(scored["model_score"]) < -0.99), "a higher score must mean a lower credit score"


def test_calibrated_probability_tracks_the_observed_bad_rate(trained):
    result, _, data = trained
    scored = result.bundle.score(data)
    predicted = scored["calibrated_prob"].mean()
    observed = data[LABEL].mean()
    assert abs(predicted - observed) < 0.03, f"predicted {predicted:.3f} vs observed {observed:.3f}"


def test_unseen_categories_do_not_break_scoring(trained):
    result, _, data = trained
    batch = data.head(100).copy()
    batch["channel"] = "a_channel_that_never_existed"
    batch.loc[batch.index[:10], "employment"] = None

    scored = result.bundle.score(batch)
    assert len(scored) == 100
    assert scored["model_score"].notna().all()


def test_missing_a_required_column_fails_clearly(trained):
    result, _, data = trained
    with pytest.raises((KeyError, ValueError), match="debt_ratio"):
        result.bundle.score(data.drop(columns=["debt_ratio"]))


def test_drift_is_quiet_on_the_training_population_and_loud_on_a_shifted_one(trained):
    from riskflow.settings import Settings

    result, _, data = trained
    monitoring = Settings().monitoring

    quiet = result.bundle.drift_report(data, monitoring)
    assert alerts(quiet, monitoring) == [], quiet.to_string()

    shifted = data.copy()
    shifted["debt_ratio"] = np.clip(shifted["debt_ratio"] + 0.35, 0, 1)
    shifted.loc[shifted.index[: len(shifted) // 3], "utilization"] = np.nan
    loud = result.bundle.drift_report(shifted, monitoring)

    assert alerts(loud, monitoring), "a 0.35 shift in a key driver must be reported"
    assert loud.loc[loud["feature"] == "debt_ratio", "level"].iloc[0] == "alert"
    assert loud.loc[loud["feature"] == "utilization", "missing_rate_shift"].iloc[0] > 0.2


def test_promote_then_score_uses_the_promoted_run(trained):
    result, root, data = trained
    registry = Registry(root)

    with pytest.raises(FileNotFoundError, match="no production run"):
        registry.production()

    registry.promote(result.run)
    assert registry.production().name == result.run.name

    outcome = score_batch(data.head(300), registry_root=root)
    assert outcome.run_name == result.run.name
    assert set(outcome.scores["decision"]) <= {"approve", "review", "reject"}


def test_promoting_an_incomplete_run_is_refused(trained, tmp_path):
    _, root, _ = trained
    registry = Registry(root)
    broken = registry.new_run("half_finished")
    with pytest.raises(ValueError, match="refusing to promote"):
        registry.promote(broken)


def test_comparing_a_run_with_itself_shows_no_change(trained):
    from riskflow.compare import compare_runs

    result, root, data = trained
    comparison = compare_runs(result.run.name, result.run.name, registry_root=root, sample=data.head(200))

    assert (comparison.metrics["delta"].abs() < 1e-9).all()
    assert comparison.decision_shift.attrs["flipped_share"] == 0.0


def test_a_bundle_from_an_unknown_format_is_rejected(trained, tmp_path):
    result, _, _ = trained
    payload = json.loads(result.run.bundle_path.read_text(encoding="utf-8"))
    payload["format_version"] = 99
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="format version 99"):
        ScoringBundle.load(path)
