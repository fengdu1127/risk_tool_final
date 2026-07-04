"""Tests for score.py: policy application, drift check, and pipeline round trip."""
import json
import numpy as np
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from score import apply_policy, compute_drift, run_scoring


@pytest.fixture
def policy():
    return {
        "score_thresholds": {"reject": 0.8, "review": 0.6},
        "segment_overrides": [
            {"feature": "channel", "value": "B", "reject_threshold": 0.5, "review_threshold": 0.4},
        ],
        "reject_rules": [
            {"rule_type": "single", "feature": "overdue_cnt", "direction": ">=", "threshold": 5,
             "condition_str": "overdue_cnt >= 5"},
        ],
    }


class TestApplyPolicy:
    def test_score_thresholds(self, policy):
        df = pd.DataFrame({"channel": ["A"] * 3, "overdue_cnt": [0, 0, 0]})
        scores = np.array([0.9, 0.7, 0.3])
        out = apply_policy(df, scores, policy)
        assert out["decision"].tolist() == ["reject", "review", "approve"]

    def test_rule_reject_overrides_low_score(self, policy):
        df = pd.DataFrame({"channel": ["A"], "overdue_cnt": [7]})
        out = apply_policy(df, np.array([0.1]), policy)
        assert out["decision"].iloc[0] == "reject"
        assert bool(out["reject_by_rule"].iloc[0])
        assert "overdue_cnt >= 5" in out["hit_rules"].iloc[0]

    def test_segment_override_threshold(self, policy):
        # score 0.55: approve for channel A (global reject 0.8) but reject for channel B (override 0.5)
        df = pd.DataFrame({"channel": ["A", "B"], "overdue_cnt": [0, 0]})
        out = apply_policy(df, np.array([0.55, 0.55]), policy)
        assert out["decision"].tolist() == ["approve", "reject"]

    def test_no_policy_returns_scores_only(self):
        df = pd.DataFrame({"x": [1, 2]})
        out = apply_policy(df, np.array([0.1, 0.9]), None)
        assert "model_score" in out.columns
        assert "decision" not in out.columns


class TestComputeDrift:
    @pytest.fixture
    def baseline(self):
        np.random.seed(0)
        vals = np.random.normal(0, 1, 2000)
        edges = np.quantile(vals, np.linspace(0, 1, 11))
        interior = [float(e) for e in np.unique(edges)[1:-1]]
        counts, _ = np.histogram(vals, bins=np.array([-np.inf] + interior + [np.inf]))
        return {"x": {
            "breakpoints": interior,
            "expected_pct": list(counts / counts.sum()),
            "missing_rate": 0.0,
        }}

    def test_no_drift_on_same_distribution(self, baseline):
        np.random.seed(1)
        df = pd.DataFrame({"x": np.random.normal(0, 1, 2000)})
        drift = compute_drift(df, baseline)
        assert drift.iloc[0]["PSI"] < 0.1
        assert drift.iloc[0]["psi_level"] == "stable"

    def test_shifted_distribution_flagged(self, baseline):
        np.random.seed(2)
        df = pd.DataFrame({"x": np.random.normal(2.5, 1, 2000)})
        drift = compute_drift(df, baseline)
        assert drift.iloc[0]["PSI"] >= 0.25
        assert drift.iloc[0]["psi_level"] == "unstable"

    def test_missing_rate_delta(self, baseline):
        np.random.seed(3)
        x = np.random.normal(0, 1, 1000)
        x[:300] = np.nan
        drift = compute_drift(pd.DataFrame({"x": x}), baseline)
        assert drift.iloc[0]["missing_rate_delta"] > 0.25

    def test_empty_baseline(self):
        assert compute_drift(pd.DataFrame({"x": [1.0]}), None).empty
        assert compute_drift(pd.DataFrame({"x": [1.0]}), {}).empty


@pytest.mark.slow
class TestEndToEnd:
    def test_pipeline_then_score(self, tmp_path):
        """Train on synthetic data with pipeline, then score new rows through score.py."""
        from pipeline import run_pipeline

        np.random.seed(42)
        n = 900
        df = pd.DataFrame({
            "score": np.random.normal(650, 80, n).clip(300, 850),
            "debt_ratio": np.random.beta(2, 5, n),
            "channel": np.random.choice(["A", "B", "C"], n),
        })
        prob_bad = 1 / (1 + np.exp((df["score"] - 550) / 80 - df["debt_ratio"] * 2))
        df["label"] = (np.random.random(n) < prob_bad).astype(int)
        # inject missing so the missing-bin path is exercised
        df.loc[df.sample(frac=0.1, random_state=1).index, "debt_ratio"] = np.nan

        data_path = tmp_path / "train.csv"
        df.to_csv(data_path, index=False)
        run_dir = tmp_path / "run"
        summary = run_pipeline(
            data_path=str(data_path),
            label_col="label",
            algo="lr",
            tune=False,
            output_dir=str(run_dir),
        )
        assert summary["best_model"] == "lr"
        assert (run_dir / "model" / "model_meta.json").exists()
        meta = json.loads((run_dir / "model" / "model_meta.json").read_text(encoding="utf-8"))
        assert "drift_baseline" in meta and "score" in meta["drift_baseline"]

        new_data = tmp_path / "new.csv"
        df.head(200).drop(columns=["label"]).to_csv(new_data, index=False)
        out_path = tmp_path / "scored.csv"
        result = run_scoring(str(run_dir), str(new_data), str(out_path))
        assert len(result) == 200
        assert out_path.exists()
        assert result["model_score"].between(0, 1).all()
        if "decision" in result.columns:
            assert set(result["decision"].unique()).issubset({"reject", "review", "approve"})
