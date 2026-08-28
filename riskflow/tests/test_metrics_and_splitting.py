from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskflow.data.schema import DatasetSchema, infer_schema, validate_frame
from riskflow.data.splitting import out_of_time_split, random_split, split
from riskflow.models import metrics as M
from riskflow.settings import SplitSettings

LABEL = "is_bad"


# --------------------------------------------------------------------- metrics


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.3).astype(int)
    # heavy rounding creates many tied scores, where rank handling matters
    scores = np.round(y * 0.4 + rng.normal(0.5, 0.2, 2000), 2)

    from sklearn.metrics import roc_auc_score

    assert M.auc(y, scores) == pytest.approx(roc_auc_score(y, scores), abs=1e-9)


def test_ks_matches_a_direct_roc_computation():
    rng = np.random.default_rng(4)
    y = (rng.random(1500) < 0.25).astype(int)
    scores = y * 0.5 + rng.normal(0, 0.3, 1500)

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, scores)
    assert M.ks(y, scores) == pytest.approx(float(np.max(tpr - fpr)), abs=1e-9)


def test_perfect_and_useless_scores_sit_at_the_expected_extremes():
    y = np.array([0, 0, 1, 1])
    assert M.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert M.ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert M.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert M.ks(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_single_class_metrics_are_nan_rather_than_misleading():
    y = np.ones(50)
    assert np.isnan(M.auc(y, np.arange(50.0)))
    assert np.isnan(M.ks(y, np.arange(50.0)))


def test_gains_table_orders_riskiest_first_and_captures_all_bads():
    rng = np.random.default_rng(5)
    y = (rng.random(2000) < 0.2).astype(int)
    scores = np.clip(y * 0.4 + rng.normal(0.3, 0.15, 2000), 0, 1)
    table = M.gains_table(y, scores, "test")

    assert table["band"].is_monotonic_decreasing
    assert table["bad_rate"].iloc[0] > table["bad_rate"].iloc[-1]
    assert table["cum_bad_capture"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
    assert table["share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_fixed_band_edges_make_samples_comparable():
    rng = np.random.default_rng(6)
    y = (rng.random(1000) < 0.2).astype(int)
    scores = rng.random(1000)
    edges = M.band_edges(scores, 10)
    # a shifted sample keeps the reference bands rather than re-quantiling itself
    shifted = M.gains_table(y, scores * 0.5, "shifted", edges=edges)
    assert shifted["rows"].sum() == 1000
    assert shifted["band"].max() < 9


# -------------------------------------------------------------------- splitting


def test_random_split_partitions_every_row_exactly_once(applications):
    parts = random_split(applications, LABEL, SplitSettings())
    total = sum(len(frame) for _, frame in parts.items())
    assert total == len(applications)
    ids = pd.concat([frame["application_id"] for _, frame in parts.items()])
    assert ids.nunique() == len(applications)


def test_random_split_preserves_the_bad_rate(applications):
    parts = random_split(applications, LABEL, SplitSettings())
    overall = applications[LABEL].mean()
    for _, frame in parts.items():
        assert frame[LABEL].mean() == pytest.approx(overall, abs=0.02)


def test_random_split_is_reproducible(applications):
    a = random_split(applications, LABEL, SplitSettings(random_state=99))
    b = random_split(applications, LABEL, SplitSettings(random_state=99))
    pd.testing.assert_frame_equal(a.holdout, b.holdout)


def test_out_of_time_holdout_is_strictly_newer(applications):
    settings = SplitSettings(time_col="apply_time", oot_months=3)
    parts = out_of_time_split(applications, LABEL, settings)
    newest_history = pd.to_datetime(
        pd.concat([parts.train["apply_time"], parts.test["apply_time"]])
    ).max()
    oldest_holdout = pd.to_datetime(parts.holdout["apply_time"]).min()
    assert oldest_holdout > newest_history
    assert parts.strategy == "out_of_time"


def test_split_dispatches_on_the_time_column(applications):
    assert split(applications, LABEL, SplitSettings()).strategy == "random_stratified"
    assert split(applications, LABEL, SplitSettings(time_col="apply_time")).strategy == "out_of_time"


def test_a_holdout_too_small_to_validate_is_refused(applications):
    with pytest.raises(ValueError, match="too small to validate"):
        out_of_time_split(
            applications, LABEL, SplitSettings(time_col="apply_time", oot_months=1, min_holdout_rows=100_000)
        )


def test_ratios_must_sum_to_one(applications):
    with pytest.raises(ValueError, match="must sum to 1"):
        random_split(applications, LABEL, SplitSettings(train_ratio=0.8, test_ratio=0.3, holdout_ratio=0.1))


# ----------------------------------------------------------------------- schema


def test_schema_separates_numeric_from_categorical(applications):
    schema = infer_schema(applications, LABEL, time_col="apply_time", id_col="application_id")
    assert "debt_ratio" in schema.numeric
    assert "channel" in schema.categorical
    for excluded in (LABEL, "apply_time", "application_id"):
        assert excluded not in schema.features


def test_schema_rejects_impossible_requests(applications):
    with pytest.raises(ValueError, match="not in data"):
        infer_schema(applications, "does_not_exist")
    with pytest.raises(ValueError, match="cannot be features"):
        infer_schema(applications, LABEL, features=[LABEL, "age"])


def test_validate_frame_raises_on_a_non_binary_label(applications):
    frame = applications.copy()
    frame[LABEL] = frame["age"]
    schema = DatasetSchema(label=LABEL, numeric=("debt_ratio",), categorical=())
    with pytest.raises(ValueError, match="must be 0/1"):
        validate_frame(frame, schema)


def test_validate_frame_reports_missing_features_by_name(applications):
    schema = DatasetSchema(label=LABEL, numeric=("nope",), categorical=())
    with pytest.raises(ValueError, match="nope"):
        validate_frame(applications, schema, require_label=False)
