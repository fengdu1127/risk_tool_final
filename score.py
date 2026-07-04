"""Batch scoring entrypoint: apply a trained run (model + WOE encoder + policy) to new data.

Usage:
    python score.py --run-dir reports/run_xxx --data new_applications.csv --output scored_output.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from modules.strategy.auto_strategy import evaluate_rule_mask
from utils.helpers import get_logger, load_pickle, prob_to_score

logger = get_logger("SCORE")

PRODUCTION_POINTER = os.path.join("reports", "PRODUCTION")


def resolve_run_dir(run_dir: str = None) -> str:
    """Explicit --run-dir wins; otherwise fall back to the reports/PRODUCTION pointer."""
    if run_dir:
        return run_dir
    if os.path.exists(PRODUCTION_POINTER):
        with open(PRODUCTION_POINTER, "r", encoding="utf-8") as f:
            pointed = f.read().strip()
        if pointed:
            logger.info(f"using production run from {PRODUCTION_POINTER}: {pointed}")
            return pointed
    raise ValueError(
        "no --run-dir given and reports/PRODUCTION does not exist; "
        "promote a run first: python promote.py reports/run_xxx"
    )


def load_artifacts(run_dir: str) -> dict:
    model_dir = os.path.join(run_dir, "model")
    strategy_dir = os.path.join(run_dir, "strategy")

    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"model_meta.json not found in {model_dir}; re-run pipeline.py to generate scoring metadata"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    best_algo = meta.get("best_algo")
    if best_algo == "xgboost":
        import xgboost as xgb

        json_path = os.path.join(model_dir, "model_xgb.json")
        if os.path.exists(json_path):
            model = xgb.XGBClassifier()
            model.load_model(json_path)
        else:
            model = load_pickle(os.path.join(model_dir, "model_xgb.pkl"))
    elif best_algo == "lr":
        model = load_pickle(os.path.join(model_dir, "model_lr.pkl"))
    else:
        raise ValueError(f"unsupported best_algo in model_meta.json: {best_algo}")

    encoder = load_pickle(os.path.join(model_dir, "woe_encoder.pkl"))

    calibrator = None
    calibrator_path = os.path.join(model_dir, "calibrator.pkl")
    if os.path.exists(calibrator_path):
        calibrator = load_pickle(calibrator_path)

    policy = None
    policy_path = os.path.join(strategy_dir, "policy.json")
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
    else:
        logger.warning("policy.json not found; output will contain scores only, no decisions")

    return {"meta": meta, "model": model, "encoder": encoder, "calibrator": calibrator, "policy": policy}


def build_features(df: pd.DataFrame, meta: dict, encoder) -> pd.DataFrame:
    feature_cols = meta["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"input data is missing model features: {missing}")

    if meta["best_algo"] == "lr":
        X = encoder.transform(df[feature_cols].copy(), feature_cols)[feature_cols].fillna(0)
        return X

    xgb_cols = meta.get("xgb_feature_cols") or feature_cols
    num_cols = [c for c in xgb_cols if c not in encoder.cat_features]
    cat_cols = [c for c in xgb_cols if c in encoder.cat_features]
    X = df[num_cols].copy()
    if cat_cols:
        encoded = encoder.transform(df[cat_cols].copy(), cat_cols)
        for c in cat_cols:
            X[c] = encoded[c].values
    return X[xgb_cols]


def compute_drift(df: pd.DataFrame, baseline: dict, floor: float = 1e-6) -> pd.DataFrame:
    """PSI + missing-rate shift of the new batch vs the training distribution."""
    rows = []
    for col, base in (baseline or {}).items():
        if col not in df.columns:
            continue
        edges = np.array([-np.inf] + list(base["breakpoints"]) + [np.inf])
        expected = np.asarray(base["expected_pct"], dtype=float)
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(vals) == 0 or len(expected) != len(edges) - 1:
            continue
        counts, _ = np.histogram(vals, bins=edges)
        actual = counts / counts.sum()
        expected_f = np.where(expected == 0, floor, expected)
        actual_f = np.where(actual == 0, floor, actual)
        psi = float(np.sum((actual_f - expected_f) * np.log(actual_f / expected_f)))
        new_missing = float(df[col].isna().mean())
        rows.append({
            "feature": col,
            "PSI": round(psi, 6),
            "psi_level": "stable" if psi < 0.1 else "warning" if psi < 0.25 else "unstable",
            "train_missing_rate": round(base.get("missing_rate", 0.0), 6),
            "new_missing_rate": round(new_missing, 6),
            "missing_rate_delta": round(new_missing - base.get("missing_rate", 0.0), 6),
        })
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def apply_policy(df: pd.DataFrame, scores: np.ndarray, policy: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["model_score"] = scores
    if not policy:
        return out

    rule_hits = pd.Series([""] * len(df), index=df.index)
    rule_reject = pd.Series(False, index=df.index)
    for rule in policy.get("reject_rules", []):
        try:
            mask = evaluate_rule_mask(df, rule).reindex(df.index).fillna(False).astype(bool)
        except Exception as exc:
            logger.warning(f"rule skipped ({rule.get('condition_str', '?')}): {exc}")
            continue
        rule_reject = rule_reject | mask
        cond = rule.get("condition_str", "")
        rule_hits = rule_hits.where(~mask, rule_hits + cond + "; ")

    reject_thr = pd.Series(np.nan, index=df.index)
    review_thr = pd.Series(np.nan, index=df.index)
    thresholds = policy.get("score_thresholds")
    if thresholds:
        reject_thr[:] = thresholds["reject"]
        review_thr[:] = thresholds["review"]
    for override in policy.get("segment_overrides", []):
        feature, value = override.get("feature"), override.get("value")
        if feature not in df.columns:
            continue
        seg_mask = df[feature].isna() if value is None else (df[feature] == value)
        reject_thr = reject_thr.where(~seg_mask, override["reject_threshold"])
        review_thr = review_thr.where(~seg_mask, override["review_threshold"])

    score_s = pd.Series(scores, index=df.index)
    score_reject = reject_thr.notna() & (score_s >= reject_thr)
    score_review = review_thr.notna() & (score_s >= review_thr) & (~score_reject)

    decision = pd.Series("approve", index=df.index)
    decision[score_review] = "review"
    decision[rule_reject | score_reject] = "reject"

    out["decision"] = decision
    out["reject_by_rule"] = rule_reject
    out["reject_by_score"] = score_reject
    out["hit_rules"] = rule_hits.str.rstrip("; ")
    out["reject_threshold"] = reject_thr
    out["review_threshold"] = review_thr
    return out


def run_scoring(run_dir: str, data_path: str, output_path: str = None, id_col: str = None) -> pd.DataFrame:
    run_dir = resolve_run_dir(run_dir)
    artifacts = load_artifacts(run_dir)
    df = pd.read_csv(data_path)
    logger.info(f"scoring {len(df)} rows with model from {run_dir} (best_algo={artifacts['meta']['best_algo']})")

    drift = compute_drift(df, artifacts["meta"].get("drift_baseline"))
    if not drift.empty:
        unstable = drift[drift["psi_level"] == "unstable"]["feature"].tolist()
        warning = drift[drift["psi_level"] == "warning"]["feature"].tolist()
        missing_shift = drift[drift["missing_rate_delta"].abs() > 0.10]["feature"].tolist()
        if unstable:
            logger.warning(f"DRIFT ALERT - unstable features (PSI>=0.25): {unstable}")
        if warning:
            logger.warning(f"drift warning (0.1<=PSI<0.25): {warning}")
        if missing_shift:
            logger.warning(f"missing-rate shift >10pp: {missing_shift}")
        if not (unstable or warning or missing_shift):
            logger.info("drift check passed: all features stable vs train")
    else:
        logger.warning("no drift baseline in model_meta.json; skipping drift check")

    X = build_features(df, artifacts["meta"], artifacts["encoder"])
    scores = artifacts["model"].predict_proba(X)[:, 1]
    result = apply_policy(df, scores, artifacts["policy"])

    if artifacts["calibrator"] is not None:
        result.insert(1, "calibrated_prob", artifacts["calibrator"].predict(scores))
    scorecard = artifacts["meta"].get("scorecard") or {}
    result.insert(1, "credit_score", np.round(prob_to_score(
        scores,
        pdo=scorecard.get("pdo", 20),
        base_score=scorecard.get("base_score", 600),
        base_odds=scorecard.get("base_odds", 1 / 15),
    ), 1))

    if id_col and id_col in df.columns:
        result.insert(0, id_col, df[id_col].values)

    output_path = output_path or os.path.join(
        os.path.dirname(data_path) or ".", "scored_output.csv"
    )
    result.to_csv(output_path, index=False)
    if not drift.empty:
        drift_path = os.path.splitext(output_path)[0] + "_drift.csv"
        drift.to_csv(drift_path, index=False)
        logger.info(f"drift report saved: {drift_path}")
    if "decision" in result.columns:
        dist = result["decision"].value_counts(normalize=True).round(4).to_dict()
        logger.info(f"decision distribution: {dist}")
    logger.info(f"saved {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new data with a trained pipeline run")
    parser.add_argument("--run-dir", default=None, help="Pipeline output directory (default: reports/PRODUCTION pointer)")
    parser.add_argument("--data", required=True, help="CSV with the same feature columns used in training")
    parser.add_argument("--output", default=None, help="Output CSV path (default: scored_output.csv next to input)")
    parser.add_argument("--id-col", default=None, help="Optional ID column to carry into the output")
    args = parser.parse_args()

    run_scoring(args.run_dir, args.data, args.output, args.id_col)
