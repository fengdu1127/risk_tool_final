from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskflow.data.schema import infer_schema
from riskflow.features.binning import fit_numeric
from riskflow.features.diagnostics import binned_psi, correlated_pairs, psi, screen, variance_inflation
from riskflow.features.woe import WoeTransformer
from riskflow.monitoring.drift import DriftBaseline, alerts
from riskflow.settings import BinningSettings, MonitoringSettings, ScreeningSettings

LABEL = "is_bad"


# ------------------------------------------------------------------------ PSI


def test_psi_is_zero_for_an_identical_population():
    rng = np.random.default_rng(41)
    sample = rng.normal(size=5000)
    assert psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=8000)
    small = psi(reference, rng.normal(0.1, 1, 8000))
    large = psi(reference, rng.normal(1.0, 1, 8000))
    assert small < 0.1 < large


def test_binned_psi_uses_the_model_s_own_bins():
    """A batch that is stable in its own quantiles can still have shifted bins."""
    rng = np.random.default_rng(43)
    x = rng.normal(size=6000)
    y = (rng.random(6000) < 1 / (1 + np.exp(-x))).astype(int)
    binning = fit_numeric(x, y, BinningSettings(), "x")

    shifted = pd.Series(rng.normal(1.2, 1.0, 6000))
    # its internal shape is unchanged, so a self-quantiled PSI sees nothing
    assert psi(shifted, shifted) == pytest.approx(0.0, abs=1e-9)
    # against the fitted bins, the move is obvious
    assert binned_psi(binning, pd.Series(x), shifted) > 0.25


# ------------------------------------------------------------------ redundancy


def test_correlated_pairs_finds_a_duplicated_column():
    rng = np.random.default_rng(44)
    base = rng.normal(size=1000)
    frame = pd.DataFrame({"a": base, "a_copy": base + rng.normal(0, 0.01, 1000), "b": rng.normal(size=1000)})
    pairs = correlated_pairs(frame, 0.7)

    assert len(pairs) == 1
    assert set(pairs.iloc[0][["feature_a", "feature_b"]]) == {"a", "a_copy"}


def test_vif_flags_a_linear_combination():
    rng = np.random.default_rng(45)
    a, b = rng.normal(size=1000), rng.normal(size=1000)
    frame = pd.DataFrame({"a": a, "b": b, "sum": a + b + rng.normal(0, 1e-3, 1000)})
    vif = variance_inflation(frame)

    assert vif["sum"] > 10
    assert np.isfinite(vif).all(), "perfect collinearity must not produce infinities"


def test_vif_is_one_for_independent_columns():
    rng = np.random.default_rng(46)
    frame = pd.DataFrame({f"v{i}": rng.normal(size=2000) for i in range(4)})
    assert variance_inflation(frame).max() < 1.2


# ------------------------------------------------------------------- screening


def test_screening_records_a_reason_for_every_feature(applications, schema, settings):
    woe = WoeTransformer.fit(applications, schema, settings.binning)
    result = screen(applications, None, schema, woe, settings.screening)

    assert set(result.report["feature"]) == set(schema.features)
    assert (result.report["reason"].str.len() > 0).all()
    assert set(result.selected) == set(result.report.loc[result.report["selected"], "feature"])


def test_screening_drops_noise_and_keeps_drivers(applications, schema, settings):
    woe = WoeTransformer.fit(applications, schema, settings.binning)
    result = screen(applications, None, schema, woe, settings.screening)

    assert "noise_score" not in result.selected
    assert "debt_ratio" in result.selected
    reason = result.report.loc[result.report["feature"] == "noise_score", "reason"].iloc[0]
    assert "IV" in reason


def test_screening_keeps_the_stronger_of_two_duplicates(applications, schema, settings):
    frame = applications.copy()
    frame["debt_ratio_copy"] = frame["debt_ratio"] * 1.0001
    duplicated_schema = infer_schema(frame, LABEL, time_col="apply_time", id_col="application_id")
    woe = WoeTransformer.fit(frame, duplicated_schema, settings.binning)
    result = screen(frame, None, duplicated_schema, woe, settings.screening)

    kept = {"debt_ratio", "debt_ratio_copy"} & set(result.selected)
    assert len(kept) == 1, "one of a duplicated pair must go"
    dropped = ({"debt_ratio", "debt_ratio_copy"} - kept).pop()
    assert "correlated with" in result.report.loc[result.report["feature"] == dropped, "reason"].iloc[0]


def test_a_leaky_feature_is_called_out(applications, schema, settings):
    frame = applications.copy()
    frame["leak"] = frame[LABEL] * 10 + np.random.default_rng(47).normal(0, 0.1, len(frame))
    leaky_schema = infer_schema(frame, LABEL, time_col="apply_time", id_col="application_id")
    woe = WoeTransformer.fit(frame, leaky_schema, settings.binning)
    result = screen(frame, None, leaky_schema, woe, settings.screening)

    assert "leak" not in result.selected
    assert "leakage" in result.report.loc[result.report["feature"] == "leak", "reason"].iloc[0]


def test_screening_that_rejects_everything_fails_loudly(applications, schema, settings):
    woe = WoeTransformer.fit(applications, schema, settings.binning)
    with pytest.raises(ValueError, match="rejected every feature"):
        screen(applications, None, schema, woe, ScreeningSettings(min_iv=99.0))


# ----------------------------------------------------------------------- drift


@pytest.fixture
def baseline(applications, schema, settings):
    woe = WoeTransformer.fit(applications, schema, settings.binning)
    features = [f for f in schema.features if f in woe.features]
    return DriftBaseline.fit(applications, woe, features, scores=np.linspace(0, 1, len(applications))), woe


def test_drift_is_silent_on_the_reference_population(applications, baseline):
    base, woe = baseline
    report = base.report(applications, woe, MonitoringSettings())
    assert (report.loc[report["feature"] != "__model_score__", "psi"] < 0.01).all()
    assert alerts(report, MonitoringSettings()) == []


def test_drift_reports_a_shifted_feature_and_a_missing_rate_jump(applications, baseline):
    base, woe = baseline
    shifted = applications.copy()
    shifted["debt_ratio"] = np.clip(shifted["debt_ratio"] + 0.4, 0, 1)
    shifted.loc[shifted.index[: len(shifted) // 2], "income"] = np.nan

    report = base.report(shifted, woe, MonitoringSettings()).set_index("feature")
    assert report.loc["debt_ratio", "level"] == "alert"
    assert report.loc["income", "missing_rate_shift"] > 0.4
    messages = alerts(report.reset_index(), MonitoringSettings())
    assert any("debt_ratio" in m for m in messages)
    assert any("missing rate" in m for m in messages)


def test_a_column_that_disappeared_is_reported_not_ignored(applications, baseline):
    base, woe = baseline
    report = base.report(applications.drop(columns=["debt_ratio"]), woe, MonitoringSettings())
    row = report[report["feature"] == "debt_ratio"].iloc[0]
    assert row["level"] == "absent"


def test_drift_baseline_round_trips_through_json(applications, baseline):
    import json

    base, woe = baseline
    restored = DriftBaseline.from_dict(json.loads(json.dumps(base.to_dict())))
    pd.testing.assert_frame_equal(
        restored.report(applications, woe, MonitoringSettings()),
        base.report(applications, woe, MonitoringSettings()),
    )
