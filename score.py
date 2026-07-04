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
from utils.helpers import get_logger, load_pickle

logger = get_logger("SCORE")


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

    policy = None
    policy_path = os.path.join(strategy_dir, "policy.json")
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
    else:
        logger.warning("policy.json not found; output will contain scores only, no decisions")

    return {"meta": meta, "model": model, "encoder": encoder, "policy": policy}


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
    artifacts = load_artifacts(run_dir)
    df = pd.read_csv(data_path)
    logger.info(f"scoring {len(df)} rows with model from {run_dir} (best_algo={artifacts['meta']['best_algo']})")

    X = build_features(df, artifacts["meta"], artifacts["encoder"])
    scores = artifacts["model"].predict_proba(X)[:, 1]
    result = apply_policy(df, scores, artifacts["policy"])

    if id_col and id_col in df.columns:
        result.insert(0, id_col, df[id_col].values)

    output_path = output_path or os.path.join(
        os.path.dirname(data_path) or ".", "scored_output.csv"
    )
    result.to_csv(output_path, index=False)
    if "decision" in result.columns:
        dist = result["decision"].value_counts(normalize=True).round(4).to_dict()
        logger.info(f"decision distribution: {dist}")
    logger.info(f"saved {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new data with a trained pipeline run")
    parser.add_argument("--run-dir", required=True, help="Pipeline output directory, e.g. reports/run_xxx")
    parser.add_argument("--data", required=True, help="CSV with the same feature columns used in training")
    parser.add_argument("--output", default=None, help="Output CSV path (default: scored_output.csv next to input)")
    parser.add_argument("--id-col", default=None, help="Optional ID column to carry into the output")
    args = parser.parse_args()

    run_scoring(args.run_dir, args.data, args.output, args.id_col)
