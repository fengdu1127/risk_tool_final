"""Automated model training for LR and XGBoost."""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from utils.helpers import (
    calc_auc,
    calc_ks,
    calc_psi,
    feature_psi_report,
    get_logger,
    model_report,
    save_pickle,
    score_bin_report,
)

logger = get_logger("MODEL")


class WOEEncoder:
    def __init__(self, bins: int = 10):
        self.bins = bins
        self.woe_maps = {}
        self.thresholds = {}
        self.cat_features = set()

    def fit(self, df: pd.DataFrame, feature_cols: list, label_col: str):
        from sklearn.tree import DecisionTreeClassifier

        for col in feature_cols:
            try:
                sub = df[[col, label_col]].dropna()
                col_data = sub[col]
                y_data = sub[label_col]
                if col_data.nunique() < 2:
                    continue

                if col_data.dtype in [object, "category"] or not pd.api.types.is_numeric_dtype(col_data):
                    grouped = sub.groupby(col)[label_col].agg(["sum", "count"])
                    grouped.columns = ["bad", "total"]
                    grouped["good"] = grouped["total"] - grouped["bad"]
                    total_bad = max(grouped["bad"].sum(), 1e-6)
                    total_good = max(grouped["good"].sum(), 1e-6)
                    grouped["bad_pct"] = (grouped["bad"] / total_bad).replace(0, 1e-6)
                    grouped["good_pct"] = (grouped["good"] / total_good).replace(0, 1e-6)
                    grouped["woe"] = np.log(grouped["good_pct"] / grouped["bad_pct"])
                    self.woe_maps[col] = dict(zip(grouped.index, grouped["woe"]))
                    self.thresholds[col] = list(grouped.index)
                    self.cat_features.add(col)
                    continue

                if col_data.nunique() < 3:
                    continue

                dt = DecisionTreeClassifier(max_leaf_nodes=self.bins, min_samples_leaf=50, random_state=42)
                dt.fit(col_data.fillna(col_data.median()).values.reshape(-1, 1), y_data)
                thresholds = sorted(set(dt.tree_.threshold[dt.tree_.threshold != -2]))
                self.thresholds[col] = thresholds

                bins = [-np.inf] + thresholds + [np.inf]
                col_bin = pd.cut(col_data, bins=bins, labels=False, duplicates="drop")
                tmp = pd.DataFrame({"bin": col_bin, "label": y_data})
                grouped = tmp.groupby("bin")["label"].agg(["sum", "count"])
                grouped.columns = ["bad", "total"]
                grouped["good"] = grouped["total"] - grouped["bad"]
                total_bad = max(grouped["bad"].sum(), 1e-6)
                total_good = max(grouped["good"].sum(), 1e-6)
                grouped["bad_pct"] = (grouped["bad"] / total_bad).replace(0, 1e-6)
                grouped["good_pct"] = (grouped["good"] / total_good).replace(0, 1e-6)
                grouped["woe"] = np.log(grouped["good_pct"] / grouped["bad_pct"])
                self.woe_maps[col] = dict(zip(grouped.index, grouped["woe"]))
            except Exception as exc:
                logger.warning(f"WOE fit failed for {col}: {exc}")
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        out = df.copy()
        for col in feature_cols:
            if col not in self.woe_maps:
                continue
            woe_map = self.woe_maps[col]
            if col in self.cat_features:
                out[col] = df[col].map(woe_map).fillna(0).values
            elif col in self.thresholds:
                thresholds = self.thresholds[col]
                bins = [-np.inf] + thresholds + [np.inf]
                col_data = df[col].fillna(df[col].median())
                bin_idx = pd.cut(col_data, bins=bins, labels=False, duplicates="drop")
                out[col] = pd.Series(bin_idx, index=df.index).map(woe_map).fillna(0).values
        return out

    def fit_transform(self, df, feature_cols, label_col):
        self.fit(df, feature_cols, label_col)
        return self.transform(df, feature_cols)


def build_scorecard(lr_model, woe_encoder: WOEEncoder, feature_cols: list, cfg: dict) -> pd.DataFrame:
    pdo = cfg.get("pdo", 20)
    base_score = cfg.get("base_score", 600)
    base_odds = cfg.get("base_odds", 1 / 15)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    clf = lr_model.named_steps["lr"] if hasattr(lr_model, "named_steps") else lr_model
    coefs = dict(zip(feature_cols, clf.coef_[0]))
    intercept = clf.intercept_[0]

    rows = []
    for feat in feature_cols:
        if feat not in woe_encoder.woe_maps:
            continue
        coef = coefs.get(feat, 0)
        for bin_id, woe in woe_encoder.woe_maps[feat].items():
            rows.append({
                "feature": feat,
                "bin": bin_id,
                "woe": round(woe, 4),
                "coef": round(coef, 4),
                "score": round(-factor * coef * woe, 2),
            })
    rows.append({"feature": "__base__", "bin": "all", "woe": 0, "coef": round(intercept, 4), "score": round(offset - factor * intercept, 2)})
    return pd.DataFrame(rows)


def plot_roc_ks(y_true, y_prob, title: str, save_path: str):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC Curve - {title}")
    axes[0].legend()
    scaled_thresholds = thresholds[::-1] / (thresholds.max() + 1e-9)
    axes[1].plot(scaled_thresholds, tpr, color="#2ecc71", lw=2, label="TPR")
    axes[1].plot(scaled_thresholds, fpr, color="#e74c3c", lw=2, label="FPR")
    axes[1].plot(scaled_thresholds, tpr - fpr, color="#9b59b6", lw=2, linestyle="--", label=f"KS={ks:.4f}")
    axes[1].set_xlabel("Score Percentile")
    axes[1].set_title(f"KS Curve - {title}")
    axes[1].legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_lift_curve(y_true, y_prob, title: str, save_path: str, n_bins: int = 10):
    report = score_bin_report(y_true, y_prob, title, n_bins=n_bins)
    if report.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(report["score_bin"].astype(str), report["bad_rate"], color="#3498db")
    axes[0].axhline(np.mean(y_true), color="red", linestyle="--", label=f"Overall={np.mean(y_true):.3f}")
    axes[0].set_xlabel("Score Bin")
    axes[0].set_ylabel("Bad Rate")
    axes[0].set_title(f"Bad Rate by Bin - {title}")
    axes[0].legend()
    axes[1].bar(report["score_bin"].astype(str), report["lift"], color="#e67e22")
    axes[1].axhline(1, color="red", linestyle="--")
    axes[1].set_xlabel("Score Bin")
    axes[1].set_ylabel("Lift")
    axes[1].set_title(f"Lift Curve - {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_shap_summary(model, X: pd.DataFrame, save_path: str, save_dir: str = "", max_display: int = 20):
    try:
        import shap

        n_bg = min(100, len(X))
        background = X.astype(float).iloc[:n_bg]
        explainer = shap.Explainer(model.predict, background)
        X_eval = X.astype(float).iloc[:min(300, len(X))]
        shap_values = explainer(X_eval)
        plt.figure(figsize=(10, max(4, min(max_display, len(X.columns)) * 0.4)))
        shap.summary_plot(shap_values, X_eval, max_display=max_display, show=False, plot_size=None)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        shap_imp = pd.DataFrame({
            "feature": X.columns.tolist(),
            "shap_importance": np.abs(shap_values.values).mean(axis=0),
        }).sort_values("shap_importance", ascending=False)
        shap_imp.to_csv(os.path.join(save_dir or os.path.dirname(save_path), "xgb_shap_importance.csv"), index=False)
    except Exception as exc:
        logger.warning(f"SHAP plotting failed: {exc}")


class AutoModel:
    def __init__(self, config: dict = None):
        from config.config import MODEL_CONFIG

        self.cfg = config or MODEL_CONFIG
        self.models = {}
        self.woe_encoder = None
        self.feature_cols = []
        self.label_col = ""
        self.best_algo = None
        self.scorecard = None
        self.eval_results = {}
        self._score_probs = {}

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: pd.DataFrame = None,
        valid_df: pd.DataFrame = None,
        label_col: str = "",
        feature_cols: list = None,
        algo: str = "both",
        tune: bool = True,
        report_dir: str = "reports/model",
    ) -> dict:
        os.makedirs(report_dir, exist_ok=True)
        if valid_df is None:
            valid_df = holdout_df
        self.label_col = label_col
        self.feature_cols = feature_cols or [c for c in train_df.columns if c != label_col]
        algos = self.cfg["algorithms"] if algo == "both" else [algo]
        logger.info(f"start model training | algorithms={algos}")

        X_train, y_train = train_df[self.feature_cols], train_df[label_col]
        X_test, y_test = test_df[self.feature_cols], test_df[label_col]
        X_valid, y_valid = valid_df[self.feature_cols], valid_df[label_col]

        all_reports = []
        if "lr" in algos:
            lr_result = self._train_lr(X_train, y_train, X_test, y_test, X_valid, y_valid, tune, report_dir)
            all_reports.extend(lr_result["reports"])
            self.models["lr"] = lr_result["model"]
            sc = build_scorecard(lr_result["model"], self.woe_encoder, self.feature_cols, self.cfg["scorecard"])
            self.scorecard = sc
            sc.to_csv(f"{report_dir}/scorecard_lr.csv", index=False)

        if "xgboost" in algos:
            xgb_result = self._train_xgboost(X_train, y_train, X_test, y_test, X_valid, y_valid, tune, report_dir)
            all_reports.extend(xgb_result["reports"])
            self.models["xgboost"] = xgb_result["model"]

        report_df = pd.DataFrame(all_reports)
        report_df.to_csv(f"{report_dir}/model_comparison.csv", index=False)
        logger.info("\n" + report_df.to_string(index=False))

        test_rows = report_df[report_df["dataset"] == "test"]
        primary_metric = self.cfg.get("primary_metric", "KS")
        if len(test_rows):
            best_row = test_rows.loc[test_rows[primary_metric].idxmax()]
            self.best_algo = best_row["model"]
            logger.info(f"best model: {self.best_algo} by test {primary_metric}={best_row[primary_metric]}")

        scored = self._score_all(train_df, test_df, valid_df, label_col, report_dir)
        self._save_enhanced_reports(train_df, test_df, valid_df, label_col, report_df, report_dir)

        self.eval_results = {
            "report_df": report_df,
            "best_algo": self.best_algo,
            "scored_datasets": scored,
            "diagnostics": self._overfit_diagnostics(report_df),
        }
        return self.eval_results

    def _train_lr(self, X_train, y_train, X_test, y_test, X_valid, y_valid, tune, report_dir):
        self.woe_encoder = WOEEncoder(bins=10)
        train_full = X_train.copy()
        train_full[self.label_col] = y_train.values
        X_train_woe = self.woe_encoder.fit_transform(train_full, self.feature_cols, self.label_col)[self.feature_cols].fillna(0)
        X_test_woe = self.woe_encoder.transform(X_test.copy(), self.feature_cols)[self.feature_cols].fillna(0)
        X_valid_woe = self.woe_encoder.transform(X_valid.copy(), self.feature_cols)[self.feature_cols].fillna(0)
        best_params = self._tune_lr(X_train_woe, y_train) if tune else self.cfg["lr_default_params"]
        pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(**best_params))])
        pipeline.fit(X_train_woe, y_train)

        reports = []
        for name, Xw, y in [("train", X_train_woe, y_train), ("test", X_test_woe, y_test), ("valid", X_valid_woe, y_valid)]:
            prob = pipeline.predict_proba(Xw)[:, 1]
            self._score_probs[("lr", name)] = prob
            row = model_report(y.values, prob, name)
            row["model"] = "lr"
            reports.append(row)
            plot_roc_ks(y.values, prob, f"LR-{name}", f"{report_dir}/lr_roc_ks_{name}.png")
            plot_lift_curve(y.values, prob, f"LR-{name}", f"{report_dir}/lr_lift_{name}.png")
        save_pickle(pipeline, f"{report_dir}/model_lr.pkl")
        save_pickle(self.woe_encoder, f"{report_dir}/woe_encoder.pkl")
        return {"model": pipeline, "reports": reports}

    def _tune_lr(self, X, y) -> dict:
        from sklearn.model_selection import cross_val_score

        def objective(trial):
            C = trial.suggest_float("C", 0.001, 10, log=True)
            clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            return cross_val_score(clf, Xs, y, cv=self.cfg["cv_folds"], scoring="roc_auc").mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.cfg["optuna_trials"])
        return {"C": study.best_params["C"], "max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced", "random_state": 42}

    def _train_xgboost(self, X_train, y_train, X_test, y_test, X_valid, y_valid, tune, report_dir):
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        X_tr, X_te, X_va = X_train[num_cols].copy(), X_test[num_cols].copy(), X_valid[num_cols].copy()
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        auto_scale_weight = float(neg / pos) if pos > 0 else 1.0
        if tune:
            best_params = self._tune_xgb(X_tr, y_train, X_te, y_test, auto_scale_weight)
        else:
            best_params = self.cfg["xgb_default_params"].copy()
            best_params["scale_pos_weight"] = auto_scale_weight
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_tr, y_train, eval_set=[(X_te, y_test)], verbose=False)

        reports = []
        for name, Xd, y in [("train", X_tr, y_train), ("test", X_te, y_test), ("valid", X_va, y_valid)]:
            prob = model.predict_proba(Xd)[:, 1]
            self._score_probs[("xgboost", name)] = prob
            row = model_report(y.values, prob, name)
            row["model"] = "xgboost"
            reports.append(row)
            plot_roc_ks(y.values, prob, f"XGB-{name}", f"{report_dir}/xgb_roc_ks_{name}.png")
            plot_lift_curve(y.values, prob, f"XGB-{name}", f"{report_dir}/xgb_lift_{name}.png")
        plot_shap_summary(model, X_te, f"{report_dir}/xgb_shap.png", report_dir)
        self._plot_feature_importance(model, X_tr.columns.tolist(), f"{report_dir}/xgb_feat_importance.png")
        save_pickle(model, f"{report_dir}/model_xgb.pkl")
        return {"model": model, "reports": reports}

    def _tune_xgb(self, X_train, y_train, X_val, y_val, scale_pos_weight: float = 1.0) -> dict:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5 * scale_pos_weight, 2.0 * scale_pos_weight),
                "random_state": 42,
                "verbosity": 0,
            }
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return calc_ks(y_val.values, clf.predict_proba(X_val)[:, 1])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.cfg["optuna_trials"])
        best = study.best_params
        best["random_state"] = 42
        best["verbosity"] = 0
        return best

    def _score_all(self, train_df, test_df, valid_df, label_col, report_dir):
        best_model = self.models.get(self.best_algo)
        if best_model is None:
            return {}
        scored = {}
        for name, df in [("train", train_df), ("test", test_df), ("valid", valid_df)]:
            df_out = df.copy()
            X = df_out[self.feature_cols]
            if self.best_algo == "lr":
                X_woe = self.woe_encoder.transform(X.copy(), self.feature_cols)[self.feature_cols].fillna(0)
                prob = best_model.predict_proba(X_woe)[:, 1]
            else:
                num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
                prob = best_model.predict_proba(X[num_cols])[:, 1]
            df_out["model_score"] = prob
            df_out["model_score_bin"] = pd.qcut(prob, q=10, labels=False, duplicates="drop")
            df_out.to_csv(f"{report_dir}/scored_{name}.csv", index=False)
            scored[name] = df_out
        return scored

    def _save_enhanced_reports(self, train_df, test_df, valid_df, label_col, report_df, report_dir):
        if not self.best_algo:
            return
        bin_frames = []
        train_prob = self._score_probs.get((self.best_algo, "train"))
        score_breakpoints = None
        if train_prob is not None:
            score_breakpoints = np.unique(np.quantile(train_prob, np.linspace(0, 1, 11)))
            if len(score_breakpoints) >= 2:
                score_breakpoints[0] = -np.inf
                score_breakpoints[-1] = np.inf
        for name, df in [("train", train_df), ("test", test_df), ("valid", valid_df)]:
            prob = self._score_probs.get((self.best_algo, name))
            if prob is not None:
                bin_frames.append(score_bin_report(df[label_col].values, prob, name, breakpoints=score_breakpoints))
        if bin_frames:
            score_bins = pd.concat(bin_frames, ignore_index=True)
            train_bins = score_bins[score_bins["dataset"] == "train"][
                ["score_bin", "bad_rate", "monotone_bad_rate"]
            ].rename(columns={
                "bad_rate": "train_bad_rate",
                "monotone_bad_rate": "train_monotone_bad_rate",
            })
            score_bins = score_bins.merge(train_bins, on="score_bin", how="left")
            score_bins["bad_rate_gap_vs_train"] = (score_bins["bad_rate"] - score_bins["train_bad_rate"]).round(6)
            score_bins["monotone_bad_rate_gap_vs_train"] = (
                score_bins["monotone_bad_rate"] - score_bins["train_monotone_bad_rate"]
            ).round(6)
            score_bins.to_csv(f"{report_dir}/score_bins.csv", index=False)
            self._score_bin_stability(score_bins).to_csv(f"{report_dir}/score_bin_stability.csv", index=False)

        stability_frames = []
        best_probs = {name: self._score_probs.get((self.best_algo, name)) for name in ["train", "test", "valid"]}
        if best_probs["train"] is not None:
            for name in ["test", "valid"]:
                if best_probs[name] is not None:
                    stability_frames.append(pd.DataFrame([{
                        "dataset": name,
                        "feature": "model_score",
                        "PSI": round(calc_psi(best_probs["train"], best_probs[name]), 6),
                    }]))
        num_features = [c for c in self.feature_cols if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c])]
        stability_frames.append(feature_psi_report(train_df, test_df, num_features, "test"))
        stability_frames.append(feature_psi_report(train_df, valid_df, num_features, "valid"))
        pd.concat([f for f in stability_frames if len(f)], ignore_index=True).to_csv(f"{report_dir}/stability_report.csv", index=False)

        diagnostics = self._overfit_diagnostics(report_df)
        pd.DataFrame([diagnostics]).to_csv(f"{report_dir}/overfit_diagnostics.csv", index=False)

    def _score_bin_stability(self, score_bins: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for dataset, sub in score_bins.groupby("dataset"):
            rows.append({
                "dataset": dataset,
                "n_bins": int(sub["score_bin"].nunique()),
                "raw_bad_rate_monotone": bool(sub["raw_bad_rate_monotone"].all()) if "raw_bad_rate_monotone" in sub else bool(sub["bad_rate_monotone"].all()),
                "monotone_bad_rate_monotone": bool(sub["bad_rate_monotone"].all()),
                "max_abs_bad_rate_gap_vs_train": round(float(sub["bad_rate_gap_vs_train"].abs().max()), 6),
                "max_abs_monotone_bad_rate_gap_vs_train": round(float(sub["monotone_bad_rate_gap_vs_train"].abs().max()), 6)
                if "monotone_bad_rate_gap_vs_train" in sub else np.nan,
            })
        return pd.DataFrame(rows)

    def _overfit_diagnostics(self, report_df):
        if not self.best_algo or report_df.empty:
            return {}
        rows = report_df[report_df["model"] == self.best_algo].set_index("dataset")
        out = {}
        if {"train", "test"}.issubset(rows.index):
            out["train_test_ks_gap"] = round(float(rows.loc["train", "KS"] - rows.loc["test", "KS"]), 6)
        if {"test", "valid"}.issubset(rows.index):
            out["test_valid_ks_gap"] = round(float(rows.loc["test", "KS"] - rows.loc["valid", "KS"]), 6)
        return out

    def _plot_feature_importance(self, model, feature_names, save_path):
        imp = model.feature_importances_
        feat_imp = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        top = feat_imp.head(30)
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
        ax.barh(top.index[::-1], top.values[::-1], color="#3498db")
        ax.set_xlabel("Feature Importance")
        ax.set_title("XGBoost Feature Importance (Top 30)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
