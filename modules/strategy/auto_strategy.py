"""Strategy rule mining with train/test/valid stability checks."""
import ast
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
warnings.filterwarnings("ignore")

from modules.eda.auto_eda import calc_woe_iv, check_monotonicity
from utils.helpers import calc_lift, calc_psi, get_logger

logger = get_logger("STRATEGY")


class VariableSelector:
    def __init__(self, config: dict):
        self.cfg = config
        self.selected = []
        self.dropped = {}
        self.selector_report = []

    def fit(self, df: pd.DataFrame, label_col: str, iv_table: pd.DataFrame = None) -> list:
        all_features = [c for c in df.columns if c != label_col]
        num_cols = [c for c in all_features if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in all_features if c not in num_cols]
        iv_map = dict(zip(iv_table["feature"], iv_table["IV"])) if iv_table is not None and len(iv_table) else {}

        num_pool = self._drop_high_corr(df, num_cols, iv_map)
        num_pool = self._filter_monotone(df, num_pool, label_col)
        num_pool = self._filter_stable(df, num_pool)
        cat_pool = [c for c in cat_cols if c in iv_map]
        for c in [c for c in cat_cols if c not in iv_map]:
            self.dropped[c] = "IV calculation unavailable"
        self.selected = num_pool + cat_pool
        logger.info(f"variable selection complete: {len(self.selected)} selected")
        return self.selected

    def _drop_high_corr(self, df, pool, iv_map):
        if len(pool) < 2:
            return pool
        corr_threshold = self.cfg["corr_threshold"]
        sub = df[pool].corr(method="spearman").abs()
        to_drop = set()
        for i in range(len(sub.columns)):
            for j in range(i + 1, len(sub.columns)):
                feat_a, feat_b = sub.columns[i], sub.columns[j]
                if feat_a in to_drop or feat_b in to_drop:
                    continue
                if sub.iloc[i, j] >= corr_threshold:
                    drop = feat_b if iv_map.get(feat_a, 0) >= iv_map.get(feat_b, 0) else feat_a
                    keep = feat_a if drop == feat_b else feat_b
                    to_drop.add(drop)
                    self.dropped[drop] = f"high correlation with {keep}: corr={sub.iloc[i, j]:.3f}"
        return [f for f in pool if f not in to_drop]

    def _filter_monotone(self, df, pool, label_col):
        min_spearman = self.cfg["monotone_spearman_min"]
        passing = []
        for feat in pool:
            try:
                wdf = calc_woe_iv(df[[feat, label_col]].dropna(), feat, label_col, bins=10, method="tree")
                mono = check_monotonicity(wdf)
                spearman = abs(mono.get("spearman", 0) or 0)
                self.selector_report.append({"feature": feat, "spearman": mono.get("spearman", np.nan), "is_monotone": mono.get("is_monotone", False)})
                if mono.get("is_monotone", False) and spearman >= min_spearman:
                    passing.append(feat)
                else:
                    self.dropped[feat] = f"non-monotone WOE: spearman={spearman:.3f}"
            except Exception as exc:
                self.dropped[feat] = f"monotonicity failed: {exc}"
        return passing

    def _filter_stable(self, df, pool):
        psi_max = self.cfg["psi_max"]
        df_shuffled = df.sample(frac=1.0, random_state=42)
        mid = len(df_shuffled) // 2
        df1, df2 = df_shuffled.iloc[:mid], df_shuffled.iloc[mid:]
        passing = []
        for feat in pool:
            try:
                psi_val = calc_psi(df1[feat].values, df2[feat].values)
                if psi_val <= psi_max:
                    passing.append(feat)
                else:
                    self.dropped[feat] = f"unstable PSI={psi_val:.3f}"
            except Exception:
                passing.append(feat)
        return passing


class SingleRuleMiner:
    def __init__(self, config: dict):
        self.cfg = config
        self.rules = pd.DataFrame()

    def mine(self, df: pd.DataFrame, label_col: str, feature_cols: list) -> pd.DataFrame:
        max_coverage = self.cfg["rule_max_coverage"]
        min_lift = self.cfg["rule_min_lift"]
        min_hit = self.cfg.get("rule_min_hit_count", 1)
        y = df[label_col].values
        overall_bad_rate = y.mean()
        n_total = len(df)
        all_rules = []

        for feat in feature_cols:
            col = df[feat].dropna()
            if len(col) == 0:
                continue
            if not pd.api.types.is_numeric_dtype(df[feat]):
                for val in col.unique():
                    mask = df[feat].astype(str) == str(val)
                    rule = self._make_rule(feat, "==", str(val), mask, y, n_total, overall_bad_rate)
                    if rule and rule["coverage"] <= max_coverage and rule["lift"] >= min_lift and rule["hit_count"] >= min_hit:
                        all_rules.append(rule)
                continue

            percentiles = np.unique(np.percentile(col, np.arange(5, 100, 5)))
            for thresh in percentiles:
                for direction in [">=", "<="]:
                    mask = df[feat] >= thresh if direction == ">=" else df[feat] <= thresh
                    rule = self._make_rule(feat, direction, round(float(thresh), 6), mask, y, n_total, overall_bad_rate)
                    if rule and rule["coverage"] <= max_coverage and rule["lift"] >= min_lift and rule["hit_count"] >= min_hit:
                        all_rules.append(rule)

        rules_df = pd.DataFrame(all_rules)
        if len(rules_df):
            rules_df = rules_df.sort_values(["feature", "direction", "lift"], ascending=[True, True, False])
            rules_df = rules_df.drop_duplicates(subset=["feature", "direction"]).sort_values("lift", ascending=False).reset_index(drop=True)
        self.rules = rules_df
        return rules_df

    def _make_rule(self, feat, direction, threshold, mask, y, n_total, overall_bad_rate):
        hit = int(mask.sum())
        if hit == 0:
            return None
        coverage = hit / n_total
        bad_rate = y[mask.values].mean()
        lift = bad_rate / overall_bad_rate if overall_bad_rate else 0.0
        return {
            "feature": feat,
            "direction": direction,
            "threshold": threshold,
            "condition": f"{feat} {direction} {threshold}",
            "coverage": round(coverage, 4),
            "hit_count": hit,
            "bad_rate": round(float(bad_rate), 4),
            "overall_bad_rate": round(float(overall_bad_rate), 4),
            "lift": round(float(lift), 4),
            "n_features": 1,
        }


class TreeRuleMiner:
    def __init__(self, config: dict):
        self.cfg = config
        self.trees = {}
        self.tree_features = []
        self.tree_rules = pd.DataFrame()

    def mine(self, df: pd.DataFrame, label_col: str, feature_cols: list) -> pd.DataFrame:
        num_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            self.tree_rules = pd.DataFrame()
            return self.tree_rules
        max_depth = self.cfg["tree_max_depth"]
        max_features = min(self.cfg["tree_max_features"], len(num_cols))
        min_leaf = self.cfg["tree_min_samples_leaf"]
        max_coverage = self.cfg["rule_max_coverage"]
        min_lift = self.cfg["rule_min_lift"]
        min_hit = self.cfg.get("rule_min_hit_count", 1)
        y = df[label_col].values
        overall_bad_rate = y.mean()
        n_total = len(df)
        X = df[num_cols].fillna(df[num_cols].median())
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=42,
        )
        tree.fit(X, y)
        self.trees["full"] = tree
        self.tree_features = num_cols
        rules = self._extract_leaf_rules(tree, num_cols, X, y, overall_bad_rate, n_total, max_coverage, min_lift, min_hit)
        self.tree_rules = pd.DataFrame(rules)
        return self.tree_rules

    def _extract_leaf_rules(self, tree, feature_cols, X, y, overall_bad_rate, n_total, max_coverage, min_lift, min_hit):
        from sklearn.tree import _tree

        tree_ = tree.tree_
        feature_name = [feature_cols[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature]
        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                leaf_mask = tree.apply(X.values) == node
                hit = int(leaf_mask.sum())
                if hit < min_hit:
                    return
                coverage = hit / n_total
                if coverage > max_coverage:
                    return
                lift = calc_lift(y, leaf_mask)
                if lift < min_lift:
                    return
                bad_rate = y[leaf_mask].mean()
                used_features = sorted({c.split(" ")[0] for c in conditions})
                rules.append({
                    "conditions": conditions.copy(),
                    "condition_str": " AND ".join(conditions),
                    "n_features": len(used_features),
                    "features_used": used_features,
                    "coverage": round(coverage, 4),
                    "hit_count": hit,
                    "bad_rate": round(float(bad_rate), 4),
                    "overall_bad_rate": round(float(overall_bad_rate), 4),
                    "lift": round(float(lift), 4),
                })
                return
            feat = feature_name[node]
            thresh = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{feat} <= {thresh:.6f}"])
            recurse(tree_.children_right[node], conditions + [f"{feat} > {thresh:.6f}"])

        recurse(0, [])
        return sorted(rules, key=lambda r: -r["lift"])

    def plot_tree(self, save_path: str):
        tree = self.trees.get("full")
        if tree is None:
            return
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(tree, feature_names=self.tree_features, filled=True, rounded=True, ax=ax, max_depth=3, fontsize=8, class_names=["Good", "Bad"])
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()


class StrategyBacktest:
    def run(self, df: pd.DataFrame, label_col: str, rules_df: pd.DataFrame, dataset_name: str = "train") -> pd.DataFrame:
        y = df[label_col].values
        n_total = len(df)
        overall_bad_rate = y.mean()
        results = []
        for idx, rule in rules_df.iterrows():
            mask = evaluate_rule_mask(df, rule)
            hit = int(mask.sum())
            if hit == 0:
                coverage = 0.0
                bad_rate = 0.0
                lift = 0.0
            else:
                coverage = hit / n_total
                bad_rate = y[mask.values].mean()
                lift = bad_rate / overall_bad_rate if overall_bad_rate else 0.0
            results.append({
                "dataset": dataset_name,
                "rule_id": idx,
                "condition": rule.get("condition_str", rule.get("condition", "")),
                "coverage": round(float(coverage), 4),
                "hit_count": hit,
                "bad_rate": round(float(bad_rate), 4),
                "lift": round(float(lift), 4),
            })
        return pd.DataFrame(results)

    def combined_effect(self, df: pd.DataFrame, label_col: str, rules_df: pd.DataFrame, dataset_name: str) -> dict:
        y = df[label_col].values
        combined_mask = pd.Series([False] * len(df), index=df.index)
        for _, rule in rules_df.iterrows():
            combined_mask = combined_mask | evaluate_rule_mask(df, rule)
        hit = int(combined_mask.sum())
        reject_rate = hit / len(df) if len(df) else 0.0
        hit_bad_rate = y[combined_mask.values].mean() if hit else 0.0
        pass_mask = ~combined_mask
        pass_bad_rate = y[pass_mask.values].mean() if pass_mask.sum() else np.nan
        overall_bad_rate = y.mean() if len(y) else np.nan
        return {
            "dataset": dataset_name,
            "rule_count": int(len(rules_df)),
            "reject_rate": round(float(reject_rate), 4),
            "hit_count": hit,
            "hit_bad_rate": round(float(hit_bad_rate), 4),
            "overall_bad_rate": round(float(overall_bad_rate), 4),
            "passed_bad_rate": round(float(pass_bad_rate), 4) if not pd.isna(pass_bad_rate) else np.nan,
            "lift": round(float(hit_bad_rate / overall_bad_rate), 4) if overall_bad_rate else 0.0,
        }


def evaluate_rule_mask(df: pd.DataFrame, rule: pd.Series | dict) -> pd.Series:
    rule_type = rule.get("rule_type", "single")
    if rule_type == "single" and rule.get("feature", "") in df.columns:
        feat = rule.get("feature")
        direction = rule.get("direction", ">=")
        threshold = rule.get("threshold")
        if direction == "==":
            return df[feat].astype(str) == str(threshold)
        threshold = float(threshold)
        if direction == ">=":
            return df[feat] >= threshold
        if direction == "<=":
            return df[feat] <= threshold

    mask = pd.Series([True] * len(df), index=df.index)
    conditions = rule.get("conditions", [])
    if isinstance(conditions, str):
        try:
            conditions = ast.literal_eval(conditions)
        except Exception:
            conditions = []
    for cond in conditions:
        parts = cond.strip().split()
        if len(parts) < 3:
            continue
        feat, op, value = parts[0], parts[1], float(parts[2])
        if feat not in df.columns:
            continue
        mask = mask & ((df[feat] <= value) if op == "<=" else (df[feat] > value))
    return mask


class AutoStrategy:
    def __init__(self, config: dict = None):
        from config.config import STRATEGY_CONFIG

        self.cfg = config or STRATEGY_CONFIG
        self.selector = VariableSelector(self.cfg)
        self.single_miner = SingleRuleMiner(self.cfg)
        self.tree_miner = TreeRuleMiner(self.cfg)
        self.backtester = StrategyBacktest()
        self.selected_vars = []
        self.single_rules = pd.DataFrame()
        self.tree_rules = pd.DataFrame()

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: pd.DataFrame = None,
        valid_df: pd.DataFrame = None,
        label_col: str = "",
        iv_table: pd.DataFrame = None,
        report_dir: str = "reports/strategy",
    ) -> dict:
        os.makedirs(report_dir, exist_ok=True)
        if valid_df is None:
            valid_df = holdout_df

        self.selected_vars = self.selector.fit(train_df, label_col, iv_table)
        if not self.selected_vars:
            logger.error("no usable variables after selection")
            return {}
        self._save_selector_report(report_dir)

        self.single_rules = self.single_miner.mine(train_df, label_col, self.selected_vars)
        self.tree_rules = self.tree_miner.mine(train_df, label_col, self.selected_vars)
        self.tree_miner.plot_tree(f"{report_dir}/decision_tree.png")

        all_rules = self._merge_rules()
        all_rules = self._drop_overlapped_rules(train_df, all_rules)
        bt_train = self.backtester.run(train_df, label_col, all_rules, "train")
        bt_test = self.backtester.run(test_df, label_col, all_rules, "test")
        bt_valid = self.backtester.run(valid_df, label_col, all_rules, "valid")
        rule_stability = self._build_rule_stability(bt_train, bt_test, bt_valid)
        stable_rules = self._stable_rules(all_rules, rule_stability)
        combined = pd.DataFrame([
            self.backtester.combined_effect(train_df, label_col, stable_rules, "train"),
            self.backtester.combined_effect(test_df, label_col, stable_rules, "test"),
            self.backtester.combined_effect(valid_df, label_col, stable_rules, "valid"),
        ])
        score_policy = self._score_policy_report(train_df, test_df, valid_df, label_col)
        marginal = self._marginal_contribution_report(train_df, test_df, valid_df, label_col, stable_rules)
        segment = self._segment_strategy_report(train_df, test_df, valid_df, label_col, stable_rules)
        overall_candidates, overall_recommendation = self._overall_strategy_search(train_df, test_df, valid_df, label_col)
        segment_discrimination = self._segment_discrimination_report(train_df, test_df, valid_df, label_col)
        filtered_segment_features = self._filter_segment_features(segment_discrimination)
        segment_candidates, segment_recommendation = self._segment_strategy_search(
            train_df,
            test_df,
            valid_df,
            label_col,
            overall_recommendation,
            filtered_segment_features,
        )
        final_comparison = self._final_strategy_comparison(
            train_df,
            test_df,
            valid_df,
            label_col,
            stable_rules,
            overall_recommendation,
            segment_recommendation,
        )
        strategy_leaderboard = self._strategy_leaderboard(final_comparison)
        strategy_recommendation = self._strategy_recommendation(strategy_leaderboard)

        self._save_outputs(
            all_rules,
            stable_rules,
            bt_train,
            bt_test,
            bt_valid,
            rule_stability,
            combined,
            score_policy,
            marginal,
            segment,
            segment_discrimination,
            overall_candidates,
            overall_recommendation,
            segment_candidates,
            segment_recommendation,
            final_comparison,
            strategy_leaderboard,
            strategy_recommendation,
            report_dir,
        )
        return {
            "selected_vars": self.selected_vars,
            "single_rules": self.single_rules,
            "tree_rules": self.tree_rules,
            "all_rules": all_rules,
            "stable_rules": stable_rules,
            "rule_stability": rule_stability,
            "backtest_train": bt_train,
            "backtest_test": bt_test,
            "backtest_valid": bt_valid,
            "combined_effect": combined,
            "score_policy": score_policy,
            "marginal_contribution": marginal,
            "segment_strategy": segment,
            "segment_discrimination": segment_discrimination,
            "filtered_segment_features": filtered_segment_features,
            "overall_strategy_candidates": overall_candidates,
            "overall_strategy_recommendation": overall_recommendation,
            "segment_strategy_candidates": segment_candidates,
            "segment_strategy_recommendation": segment_recommendation,
            "final_strategy_comparison": final_comparison,
            "strategy_leaderboard": strategy_leaderboard,
            "strategy_recommendation": strategy_recommendation,
        }

    def _merge_rules(self) -> pd.DataFrame:
        frames = []
        if len(self.single_rules):
            sr = self.single_rules.copy()
            sr["rule_type"] = "single"
            sr["condition_str"] = sr["condition"]
            frames.append(sr)
        if len(self.tree_rules):
            tr = self.tree_rules.copy()
            tr["rule_type"] = "tree"
            frames.append(tr)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("lift", ascending=False).reset_index(drop=True)

    def _drop_overlapped_rules(self, df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
        if len(rules_df) <= 1:
            return rules_df
        keep = []
        masks = []
        threshold = self.cfg.get("rule_overlap_threshold", 0.8)
        for idx, rule in rules_df.iterrows():
            mask = evaluate_rule_mask(df, rule)
            duplicate = False
            for kept_mask in masks:
                denom = min(int(mask.sum()), int(kept_mask.sum()))
                overlap = int((mask & kept_mask).sum()) / denom if denom else 0.0
                if overlap >= threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(idx)
                masks.append(mask)
        return rules_df.loc[keep].reset_index(drop=True)

    def _build_rule_stability(self, bt_train, bt_test, bt_valid):
        combined = pd.concat([bt_train, bt_test, bt_valid], ignore_index=True)
        if combined.empty:
            return combined
        wide = combined.pivot(index="rule_id", columns="dataset", values=["coverage", "bad_rate", "lift", "hit_count"])
        wide.columns = [f"{metric}_{dataset}" for metric, dataset in wide.columns]
        wide = wide.reset_index()
        rows = combined[combined["dataset"] == "train"][["rule_id", "condition"]].drop_duplicates("rule_id")
        out = rows.merge(wide, on="rule_id", how="left")
        max_cov = self.cfg["rule_max_coverage"]
        min_lift = self.cfg["rule_min_lift"]
        min_hit = self.cfg.get("rule_min_hit_count", 1)
        max_drop = self.cfg.get("rule_max_lift_drop", 0.35)
        out["lift_gap_test_valid"] = out.get("lift_test", np.nan) - out.get("lift_valid", np.nan)
        checks = []
        for _, row in out.iterrows():
            ok = True
            for ds in ["train", "test", "valid"]:
                ok &= row.get(f"coverage_{ds}", np.inf) <= max_cov
                ok &= row.get(f"lift_{ds}", -np.inf) >= min_lift
                ok &= row.get(f"hit_count_{ds}", 0) >= min_hit
            ok &= row.get("lift_gap_test_valid", 0) <= max_drop
            checks.append(bool(ok))
        out["stable_pass"] = checks
        return out

    def _stable_rules(self, all_rules, rule_stability):
        if all_rules.empty or rule_stability.empty:
            return all_rules.iloc[0:0].copy()
        stable_ids = rule_stability[rule_stability["stable_pass"]]["rule_id"].tolist()
        return all_rules.loc[stable_ids].reset_index(drop=True)

    def _save_selector_report(self, report_dir):
        report = pd.DataFrame(
            [{"feature": f, "status": "selected"} for f in self.selected_vars]
            + [{"feature": f, "status": f"dropped: {r}"} for f, r in self.selector.dropped.items()]
        )
        report.to_csv(f"{report_dir}/variable_selection.csv", index=False)

    def _score_policy_report(self, train_df, test_df, valid_df, label_col) -> pd.DataFrame:
        if "model_score" not in train_df.columns:
            return pd.DataFrame()
        reject_q = self.cfg.get("score_reject_quantile", 0.95)
        review_q = self.cfg.get("score_review_quantile", 0.85)
        reject_threshold = float(train_df["model_score"].quantile(reject_q))
        review_threshold = float(train_df["model_score"].quantile(review_q))
        rows = []
        for dataset, df in [("train", train_df), ("test", test_df), ("valid", valid_df)]:
            if "model_score" not in df.columns:
                continue
            overall_bad = df[label_col].mean()
            action_masks = {
                "reject": df["model_score"] >= reject_threshold,
                "review": (df["model_score"] >= review_threshold) & (df["model_score"] < reject_threshold),
                "approve": df["model_score"] < review_threshold,
            }
            for action, mask in action_masks.items():
                hit = int(mask.sum())
                bad_rate = df.loc[mask, label_col].mean() if hit else 0.0
                rows.append({
                    "dataset": dataset,
                    "action": action,
                    "threshold_low": review_threshold if action == "review" else reject_threshold if action == "reject" else np.nan,
                    "threshold_high": reject_threshold if action == "review" else np.nan,
                    "count": hit,
                    "rate": round(hit / len(df), 4) if len(df) else 0.0,
                    "bad_rate": round(float(bad_rate), 4),
                    "overall_bad_rate": round(float(overall_bad), 4),
                    "lift": round(float(bad_rate / overall_bad), 4) if overall_bad else 0.0,
                })
        return pd.DataFrame(rows)

    def _marginal_contribution_report(self, train_df, test_df, valid_df, label_col, stable_rules) -> pd.DataFrame:
        if stable_rules.empty:
            return pd.DataFrame()
        rows = []
        for dataset, df in [("train", train_df), ("test", test_df), ("valid", valid_df)]:
            y = df[label_col].values
            overall_bad = y.mean() if len(y) else 0.0
            cumulative_mask = pd.Series([False] * len(df), index=df.index)
            for rule_id, rule in stable_rules.iterrows():
                rule_mask = evaluate_rule_mask(df, rule)
                marginal_mask = rule_mask & (~cumulative_mask)
                marginal_hit = int(marginal_mask.sum())
                marginal_bad = int(df.loc[marginal_mask, label_col].sum()) if marginal_hit else 0
                marginal_bad_rate = marginal_bad / marginal_hit if marginal_hit else 0.0
                cumulative_mask = cumulative_mask | rule_mask
                cumulative_hit = int(cumulative_mask.sum())
                rows.append({
                    "dataset": dataset,
                    "rule_id": rule_id,
                    "condition": rule.get("condition_str", rule.get("condition", "")),
                    "rule_hit_count": int(rule_mask.sum()),
                    "marginal_hit_count": marginal_hit,
                    "marginal_bad_count": marginal_bad,
                    "marginal_bad_rate": round(float(marginal_bad_rate), 4),
                    "marginal_lift": round(float(marginal_bad_rate / overall_bad), 4) if overall_bad else 0.0,
                    "cumulative_hit_count": cumulative_hit,
                    "cumulative_reject_rate": round(cumulative_hit / len(df), 4) if len(df) else 0.0,
                })
        return pd.DataFrame(rows)

    def _segment_strategy_report(self, train_df, test_df, valid_df, label_col, stable_rules) -> pd.DataFrame:
        if stable_rules.empty:
            return pd.DataFrame()
        segment_features = self.cfg.get("segment_features", [])
        min_samples = self.cfg.get("segment_min_samples", 50)
        rows = []
        for dataset, df in [("train", train_df), ("test", test_df), ("valid", valid_df)]:
            combined_mask = pd.Series([False] * len(df), index=df.index)
            for _, rule in stable_rules.iterrows():
                combined_mask = combined_mask | evaluate_rule_mask(df, rule)
            for feature in segment_features:
                if feature not in df.columns:
                    continue
                for value, idx in df.groupby(feature, dropna=False).groups.items():
                    segment_idx = list(idx)
                    n = len(segment_idx)
                    if n < min_samples:
                        continue
                    seg_mask = pd.Series(False, index=df.index)
                    seg_mask.loc[segment_idx] = True
                    hit_mask = seg_mask & combined_mask
                    hit = int(hit_mask.sum())
                    segment_bad_rate = df.loc[seg_mask, label_col].mean()
                    hit_bad_rate = df.loc[hit_mask, label_col].mean() if hit else 0.0
                    rows.append({
                        "dataset": dataset,
                        "segment_feature": feature,
                        "segment_value": value,
                        "segment_count": n,
                        "segment_bad_rate": round(float(segment_bad_rate), 4),
                        "reject_count": hit,
                        "reject_rate": round(hit / n, 4) if n else 0.0,
                        "reject_bad_rate": round(float(hit_bad_rate), 4),
                        "lift_vs_segment": round(float(hit_bad_rate / segment_bad_rate), 4) if segment_bad_rate else 0.0,
                    })
        return pd.DataFrame(rows)

    def _segment_discrimination_report(self, train_df, test_df, valid_df, label_col) -> pd.DataFrame:
        rows = []
        datasets = {"train": train_df, "test": test_df, "valid": valid_df}
        min_share = self.cfg.get("segment_min_share", 0.10)
        max_share = self.cfg.get("segment_max_share", 0.60)
        max_share_gap = self.cfg.get("segment_max_share_gap", 0.45)
        min_bad_gap = self.cfg.get("segment_min_bad_rate_gap", 0.03)
        require_stable_order = self.cfg.get("segment_require_stable_order", False)

        for feature in self.cfg.get("segment_features", []):
            if feature not in train_df.columns:
                continue
            row = {"segment_feature": feature}
            orders = {}
            available = True
            for dataset, df in datasets.items():
                if df is None or len(df) == 0 or feature not in df.columns:
                    available = False
                    continue
                stat = (
                    df.groupby(feature, dropna=False)[label_col]
                    .agg(segment_count="count", bad_rate="mean")
                    .reset_index()
                    .sort_values(feature)
                )
                stat["share"] = stat["segment_count"] / len(df)
                row[f"{dataset}_n_segments"] = int(len(stat))
                row[f"{dataset}_min_share"] = round(float(stat["share"].min()), 6) if len(stat) else 0.0
                row[f"{dataset}_max_share"] = round(float(stat["share"].max()), 6) if len(stat) else 0.0
                row[f"{dataset}_share_gap"] = round(float(stat["share"].max() - stat["share"].min()), 6) if len(stat) else 0.0
                row[f"{dataset}_min_bad_rate"] = round(float(stat["bad_rate"].min()), 6) if len(stat) else 0.0
                row[f"{dataset}_max_bad_rate"] = round(float(stat["bad_rate"].max()), 6) if len(stat) else 0.0
                row[f"{dataset}_bad_rate_gap"] = round(float(stat["bad_rate"].max() - stat["bad_rate"].min()), 6) if len(stat) else 0.0
                row[f"{dataset}_segment_values"] = "|".join(map(str, stat[feature].tolist()))
                orders[dataset] = stat.sort_values("bad_rate", ascending=False)[feature].astype(str).tolist()

            stable_order = (
                bool(orders)
                and orders.get("train", [])[:1] == orders.get("test", [])[:1] == orders.get("valid", [])[:1]
            )
            share_pass = available and row.get("train_n_segments", 0) >= 2
            risk_pass = available
            for dataset in ["train", "test", "valid"]:
                share_pass &= row.get(f"{dataset}_min_share", 0.0) >= min_share
                share_pass &= row.get(f"{dataset}_max_share", 1.0) <= max_share
                share_pass &= row.get(f"{dataset}_share_gap", 1.0) <= max_share_gap
                risk_pass &= row.get(f"{dataset}_bad_rate_gap", 0.0) >= min_bad_gap
            stable_order_pass = stable_order or not require_stable_order
            row["stable_order"] = stable_order
            row["share_pass"] = bool(share_pass)
            row["bad_rate_gap_pass"] = bool(risk_pass)
            row["stable_order_pass"] = bool(stable_order_pass)
            row["recommend_as_segment_feature"] = bool(share_pass and risk_pass and stable_order_pass)
            rows.append(row)
        return pd.DataFrame(rows)

    def _filter_segment_features(self, discrimination_df) -> list:
        segment_features = self.cfg.get("segment_features", [])
        if not self.cfg.get("segment_use_discrimination_filter", True):
            return segment_features
        if discrimination_df is None or discrimination_df.empty:
            return []
        passed = discrimination_df[
            discrimination_df.get("recommend_as_segment_feature", False) == True
        ]["segment_feature"].tolist()
        return [feature for feature in segment_features if feature in passed]

    def _overall_strategy_search(self, train_df, test_df, valid_df, label_col) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "model_score" not in train_df.columns:
            return pd.DataFrame(), pd.DataFrame()
        rows = []
        for reject_rate in self.cfg.get("overall_reject_rate_grid", [0.05]):
            reject_threshold = float(train_df["model_score"].quantile(1 - reject_rate))
            for review_rate in self.cfg.get("overall_review_rate_grid", [0.10]):
                review_threshold = float(train_df["model_score"].quantile(max(0, 1 - reject_rate - review_rate)))
                policy_id = f"global_r{reject_rate:.3f}_v{review_rate:.3f}"
                rows.extend(self._evaluate_score_policy(
                    policy_id,
                    "global",
                    train_df,
                    test_df,
                    valid_df,
                    label_col,
                    reject_threshold,
                    review_threshold,
                ))
        candidates = pd.DataFrame(rows)
        recommendation = self._recommend_overall_strategy(candidates)
        return candidates, recommendation

    def _recommend_overall_strategy(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame()
        reject_rows = candidates[(candidates["dataset"] == "test") & (candidates["action"] == "reject")].copy()
        if reject_rows.empty:
            return pd.DataFrame()
        reject_rows = reject_rows.sort_values(
            ["lift", "bad_rate", "rate"],
            ascending=[False, False, True],
        )
        best_policy_id = reject_rows.iloc[0]["policy_id"]
        return candidates[candidates["policy_id"] == best_policy_id].copy().reset_index(drop=True)

    def _segment_strategy_search(
        self,
        train_df,
        test_df,
        valid_df,
        label_col,
        overall_recommendation,
        segment_features=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "model_score" not in train_df.columns:
            return pd.DataFrame(), pd.DataFrame()
        if segment_features is None:
            segment_features = self.cfg.get("segment_features", [])
        min_samples = self.cfg.get("segment_min_samples", 50)
        rows = []
        for feature in segment_features:
            if feature not in train_df.columns:
                continue
            for segment_value, train_seg in self._iter_segment_slices(train_df, feature):
                test_seg = self._segment_slice(test_df, feature, segment_value)
                valid_seg = self._segment_slice(valid_df, feature, segment_value)
                if len(train_seg) < min_samples or len(valid_seg) < min_samples:
                    continue
                for reject_rate in self.cfg.get("segment_reject_rate_grid", [0.05]):
                    reject_threshold = float(train_seg["model_score"].quantile(1 - reject_rate))
                    review_threshold = self._overall_review_threshold(overall_recommendation, train_df)
                    policy_id = f"segment_{feature}_{segment_value}_r{reject_rate:.3f}"
                    policy_rows = self._evaluate_score_policy(
                        policy_id,
                        "segment",
                        train_seg,
                        test_seg,
                        valid_seg,
                        label_col,
                        reject_threshold,
                        review_threshold,
                    )
                    for row in policy_rows:
                        row["segment_feature"] = feature
                        row["segment_value"] = segment_value
                    rows.extend(policy_rows)
        candidates = pd.DataFrame(rows)
        recommendation = self._recommend_segment_strategy(candidates)
        return candidates, recommendation

    def _iter_segment_slices(self, df: pd.DataFrame, feature: str):
        non_missing = df[df[feature].notna()]
        for segment_value, segment_df in non_missing.groupby(feature):
            yield segment_value, segment_df
        missing = df[df[feature].isna()]
        if len(missing):
            yield np.nan, missing

    def _segment_slice(self, df: pd.DataFrame, feature: str, segment_value) -> pd.DataFrame:
        if df is None or feature not in df.columns:
            return pd.DataFrame()
        if pd.isna(segment_value):
            return df[df[feature].isna()]
        return df[df[feature] == segment_value]

    def _recommend_segment_strategy(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame()
        valid_reject = candidates[(candidates["dataset"] == "valid") & (candidates["action"] == "reject")].copy()
        test_reject = candidates[(candidates["dataset"] == "test") & (candidates["action"] == "reject")][
            ["policy_id", "lift"]
        ].rename(columns={"lift": "test_lift"})
        valid_reject = valid_reject.merge(test_reject, on="policy_id", how="left")
        valid_reject["lift_gap_test_valid"] = valid_reject["test_lift"] - valid_reject["lift"]
        valid_reject["abs_lift_gap_test_valid"] = valid_reject["lift_gap_test_valid"].abs()
        valid_reject["recommend_segment_strategy"] = (
            (valid_reject["count"] >= self.cfg.get("segment_min_valid_hits", 10))
            & (valid_reject["lift"] >= self.cfg.get("segment_min_lift", 2.0))
            & (valid_reject["abs_lift_gap_test_valid"] <= self.cfg.get("segment_max_lift_gap", 0.5))
        )
        winners = (
            valid_reject.sort_values(
                ["segment_feature", "segment_value", "recommend_segment_strategy", "lift", "bad_rate"],
                ascending=[True, True, False, False, False],
            )
            .drop_duplicates(["segment_feature", "segment_value"])
        )
        policy_ids = winners["policy_id"].tolist()
        rec = candidates[candidates["policy_id"].isin(policy_ids)].copy()
        rec = rec.merge(
            winners[["policy_id", "recommend_segment_strategy", "lift_gap_test_valid", "abs_lift_gap_test_valid"]],
            on="policy_id",
            how="left",
        )
        rec["fallback_to_global"] = ~rec["recommend_segment_strategy"].fillna(False)
        return rec.reset_index(drop=True)

    def _final_strategy_comparison(
        self,
        train_df,
        test_df,
        valid_df,
        label_col,
        stable_rules,
        overall_recommendation,
        segment_recommendation,
    ) -> pd.DataFrame:
        rows = []
        for dataset, df in [("train", train_df), ("test", test_df), ("valid", valid_df)]:
            rows.append(self._summarize_strategy_mask(
                "global_strategy",
                dataset,
                df,
                label_col,
                self._overall_strategy_mask(df, overall_recommendation, dataset, "reject"),
                self._overall_strategy_mask(df, overall_recommendation, dataset, "review"),
            ))
            rows.append(self._summarize_strategy_mask(
                "segment_strategy",
                dataset,
                df,
                label_col,
                self._segment_strategy_mask(df, segment_recommendation, dataset, "reject"),
                self._segment_strategy_mask(df, segment_recommendation, dataset, "review"),
            ))
            mixed_reject_mask = self._overall_strategy_mask(df, overall_recommendation, dataset, "reject")
            mixed_review_mask = self._overall_strategy_mask(df, overall_recommendation, dataset, "review")
            segment_reject_mask = self._segment_strategy_mask(df, segment_recommendation, dataset, "reject")
            segment_review_mask = self._segment_strategy_mask(df, segment_recommendation, dataset, "review")
            for _, rule in stable_rules.iterrows():
                mixed_reject_mask = mixed_reject_mask | evaluate_rule_mask(df, rule)
            mixed_review_mask = (mixed_review_mask | segment_review_mask) & (~(mixed_reject_mask | segment_reject_mask))
            rows.append(self._summarize_strategy_mask(
                "global_plus_segment_rules",
                dataset,
                df,
                label_col,
                mixed_reject_mask | segment_reject_mask,
                mixed_review_mask,
            ))
        return pd.DataFrame(rows)

    def _strategy_leaderboard(self, final_comparison_df: pd.DataFrame) -> pd.DataFrame:
        if final_comparison_df is None or final_comparison_df.empty:
            return pd.DataFrame()
        rows = []
        for strategy_name, group in final_comparison_df.groupby("strategy_name"):
            row = {"strategy_name": strategy_name}
            for dataset in ["train", "test", "valid"]:
                ds = group[group["dataset"] == dataset]
                if ds.empty:
                    continue
                first = ds.iloc[0]
                for metric in [
                    "reject_rate",
                    "review_rate",
                    "approve_rate",
                    "reject_bad_rate",
                    "approve_bad_rate",
                    "overall_bad_rate",
                    "lift",
                ]:
                    row[f"{dataset}_{metric}"] = first.get(metric, np.nan)
            row["train_test_lift_gap"] = row.get("train_lift", 0.0) - row.get("test_lift", 0.0)
            row["test_valid_lift_gap"] = row.get("test_lift", 0.0) - row.get("valid_lift", 0.0)
            row["valid_stability"] = 1 / (1 + abs(row.get("test_valid_lift_gap", 0.0)))
            row["leaderboard_score"] = round(
                float(
                    row.get("valid_lift", 0.0)
                    + row.get("valid_reject_bad_rate", 0.0)
                    - abs(row.get("test_valid_lift_gap", 0.0))
                    - 0.2 * row.get("valid_reject_rate", 0.0)
                ),
                6,
            )
            rows.append(row)
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(
            ["leaderboard_score", "valid_lift", "valid_reject_rate"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    def _strategy_recommendation(self, leaderboard_df: pd.DataFrame) -> pd.DataFrame:
        if leaderboard_df is None or leaderboard_df.empty:
            return pd.DataFrame()

        def pick(frame, sort_cols, ascending):
            candidates = frame.copy()
            if candidates.empty:
                candidates = leaderboard_df.copy()
            return candidates.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()

        min_lift = self.cfg.get("segment_min_lift", 2.0)
        valid_lift = leaderboard_df.get("valid_lift", pd.Series([0] * len(leaderboard_df), index=leaderboard_df.index))
        qualified = leaderboard_df[valid_lift >= min_lift].copy()
        rows = []
        conservative = pick(
            qualified,
            ["valid_reject_rate", "test_valid_lift_gap", "valid_lift"],
            [True, True, False],
        )
        balanced = pick(
            qualified,
            ["leaderboard_score", "valid_lift", "valid_reject_rate"],
            [False, False, True],
        )
        aggressive = pick(
            qualified,
            ["valid_lift", "valid_reject_rate", "leaderboard_score"],
            [False, False, False],
        )
        for recommendation_type, chosen in [
            ("conservative", conservative),
            ("balanced", balanced),
            ("aggressive", aggressive),
        ]:
            chosen = chosen.copy()
            chosen["recommendation_type"] = recommendation_type
            rows.append(chosen)
        columns = ["recommendation_type"] + [c for c in leaderboard_df.columns if c != "recommendation_type"]
        return pd.DataFrame(rows)[columns]

    def _evaluate_score_policy(
        self,
        policy_id,
        policy_type,
        train_df,
        test_df,
        valid_df,
        label_col,
        reject_threshold,
        review_threshold,
    ) -> list[dict]:
        rows = []
        for dataset, df in [("train", train_df), ("test", test_df), ("valid", valid_df)]:
            if df is None or len(df) == 0 or "model_score" not in df.columns:
                continue
            masks = {
                "reject": df["model_score"] >= reject_threshold,
                "review": (df["model_score"] >= review_threshold) & (df["model_score"] < reject_threshold),
                "approve": df["model_score"] < review_threshold,
            }
            overall_bad = df[label_col].mean()
            for action, mask in masks.items():
                hit = int(mask.sum())
                bad_rate = df.loc[mask, label_col].mean() if hit else 0.0
                rows.append({
                    "policy_id": policy_id,
                    "policy_type": policy_type,
                    "dataset": dataset,
                    "action": action,
                    "reject_threshold": reject_threshold,
                    "review_threshold": review_threshold,
                    "count": hit,
                    "rate": round(hit / len(df), 6),
                    "bad_rate": round(float(bad_rate), 6),
                    "overall_bad_rate": round(float(overall_bad), 6),
                    "lift": round(float(bad_rate / overall_bad), 6) if overall_bad else 0.0,
                })
        return rows

    def _overall_review_threshold(self, overall_recommendation, train_df):
        if not overall_recommendation.empty and "review_threshold" in overall_recommendation.columns:
            return float(overall_recommendation["review_threshold"].iloc[0])
        return float(train_df["model_score"].quantile(self.cfg.get("score_review_quantile", 0.85)))

    def _overall_strategy_mask(self, df, overall_recommendation, dataset, action="reject"):
        if overall_recommendation.empty or "model_score" not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        row = overall_recommendation[
            (overall_recommendation["dataset"] == dataset)
            & (overall_recommendation["action"] == action)
        ]
        if row.empty:
            row = overall_recommendation.iloc[[0]]
        reject_threshold = float(row["reject_threshold"].iloc[0])
        review_threshold = float(row["review_threshold"].iloc[0])
        if action == "reject":
            return df["model_score"] >= reject_threshold
        if action == "review":
            return (df["model_score"] >= review_threshold) & (df["model_score"] < reject_threshold)
        return df["model_score"] < review_threshold

    def _segment_strategy_mask(self, df, segment_recommendation, dataset, action="reject"):
        mask = pd.Series([False] * len(df), index=df.index)
        if segment_recommendation.empty or "model_score" not in df.columns:
            return mask
        rows = segment_recommendation[
            (segment_recommendation["dataset"] == dataset)
            & (segment_recommendation["action"] == action)
            & (segment_recommendation.get("recommend_segment_strategy", False) == True)
        ]
        for _, row in rows.iterrows():
            feature = row.get("segment_feature")
            value = row.get("segment_value")
            if feature in df.columns:
                segment_mask = df[feature].isna() if pd.isna(value) else df[feature] == value
                reject_threshold = float(row["reject_threshold"])
                review_threshold = float(row["review_threshold"])
                if action == "reject":
                    action_mask = df["model_score"] >= reject_threshold
                elif action == "review":
                    action_mask = (df["model_score"] >= review_threshold) & (df["model_score"] < reject_threshold)
                else:
                    action_mask = df["model_score"] < review_threshold
                mask = mask | (segment_mask & action_mask)
        return mask

    def _summarize_strategy_mask(self, strategy_name, dataset, df, label_col, reject_mask, review_mask=None):
        reject_mask = reject_mask.reindex(df.index).fillna(False).astype(bool)
        if review_mask is None:
            review_mask = pd.Series([False] * len(df), index=df.index)
        review_mask = review_mask.reindex(df.index).fillna(False).astype(bool) & (~reject_mask)
        reject_count = int(reject_mask.sum())
        review_count = int(review_mask.sum())
        approve_mask = ~(reject_mask | review_mask)
        overall_bad = df[label_col].mean() if len(df) else 0.0
        reject_bad = df.loc[reject_mask, label_col].mean() if reject_count else 0.0
        approve_bad = df.loc[approve_mask, label_col].mean() if approve_mask.sum() else np.nan
        return {
            "strategy_name": strategy_name,
            "dataset": dataset,
            "reject_count": reject_count,
            "reject_rate": round(reject_count / len(df), 6) if len(df) else 0.0,
            "review_count": review_count,
            "review_rate": round(review_count / len(df), 6) if len(df) else 0.0,
            "approve_rate": round(int(approve_mask.sum()) / len(df), 6) if len(df) else 0.0,
            "reject_bad_rate": round(float(reject_bad), 6),
            "approve_bad_rate": round(float(approve_bad), 6) if not pd.isna(approve_bad) else np.nan,
            "overall_bad_rate": round(float(overall_bad), 6),
            "lift": round(float(reject_bad / overall_bad), 6) if overall_bad else 0.0,
        }

    def _save_outputs(
        self,
        all_rules,
        stable_rules,
        bt_train,
        bt_test,
        bt_valid,
        rule_stability,
        combined,
        score_policy,
        marginal,
        segment,
        segment_discrimination,
        overall_candidates,
        overall_recommendation,
        segment_candidates,
        segment_recommendation,
        final_comparison,
        strategy_leaderboard,
        strategy_recommendation,
        report_dir,
    ):
        all_rules.to_csv(f"{report_dir}/rules_all.csv", index=False)
        stable_rules.to_csv(f"{report_dir}/rules_stable.csv", index=False)
        pd.concat([bt_train, bt_test, bt_valid], ignore_index=True).to_csv(f"{report_dir}/backtest_results.csv", index=False)
        rule_stability.to_csv(f"{report_dir}/rule_stability.csv", index=False)
        combined.to_csv(f"{report_dir}/combined_strategy.csv", index=False)
        score_policy.to_csv(f"{report_dir}/score_policy.csv", index=False)
        marginal.to_csv(f"{report_dir}/marginal_contribution.csv", index=False)
        segment.to_csv(f"{report_dir}/segment_strategy.csv", index=False)
        segment_discrimination.to_csv(f"{report_dir}/segment_discrimination.csv", index=False)
        overall_candidates.to_csv(f"{report_dir}/overall_strategy_candidates.csv", index=False)
        overall_recommendation.to_csv(f"{report_dir}/overall_strategy_recommendation.csv", index=False)
        segment_candidates.to_csv(f"{report_dir}/segment_strategy_candidates.csv", index=False)
        segment_recommendation.to_csv(f"{report_dir}/segment_strategy_recommendation.csv", index=False)
        final_comparison.to_csv(f"{report_dir}/final_strategy_comparison.csv", index=False)
        strategy_leaderboard.to_csv(f"{report_dir}/strategy_leaderboard.csv", index=False)
        strategy_recommendation.to_csv(f"{report_dir}/strategy_recommendation.csv", index=False)
        self._plot_rule_comparison(bt_train, bt_test, bt_valid, report_dir)
        self._plot_coverage_lift(all_rules, report_dir)

    def _plot_rule_comparison(self, bt_train, bt_test, bt_valid, report_dir):
        if len(bt_train) == 0:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric, title in zip(axes, ["lift", "bad_rate"], ["Rule Lift Comparison", "Rule Bad Rate Comparison"]):
            x = np.arange(len(bt_train))
            w = 0.28
            for i, (bt, label, color) in enumerate([
                (bt_train, "Train", "#3498db"),
                (bt_test, "Test", "#e67e22"),
                (bt_valid, "Valid", "#2ecc71"),
            ]):
                if len(bt) and metric in bt.columns:
                    ax.bar(x + i * w, bt[metric].values[:len(bt_train)], width=w, label=label, color=color, alpha=0.8)
            if metric == "lift":
                ax.axhline(self.cfg["rule_min_lift"], color="red", linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Rule ID")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{report_dir}/rule_comparison.png", dpi=120)
        plt.close()

    def _plot_coverage_lift(self, all_rules, report_dir):
        if len(all_rules) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for rtype in all_rules.get("rule_type", pd.Series()).unique():
            sub = all_rules[all_rules["rule_type"] == rtype]
            ax.scatter(sub["coverage"], sub["lift"], label=rtype, alpha=0.7, s=60)
        ax.axvline(self.cfg["rule_max_coverage"], color="red", linestyle="--", linewidth=1)
        ax.axhline(self.cfg["rule_min_lift"], color="orange", linestyle="--", linewidth=1)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Lift")
        ax.set_title("Rule Coverage vs Lift")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{report_dir}/coverage_lift_scatter.png", dpi=120)
        plt.close()
