"""
modules/eda/auto_eda.py  —  自动化数据分析模块
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.binning import tree_bin_thresholds, woe_stats
from utils.helpers import get_logger, calc_psi

warnings.filterwarnings("ignore", category=FutureWarning)
logger = get_logger("EDA")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# WOE / IV 计算
# ============================================================
def calc_woe_iv(df: pd.DataFrame, feature: str, label: str,
                bins: int = 10, method: str = "quantile") -> pd.DataFrame:
    """
    计算单变量 WOE 和 IV。
    method: 'quantile'（等频）| 'uniform'（等距）| 'tree'（决策树分箱）
    """
    col = df[feature].copy()
    y = df[label].copy()

    # 分箱
    bin_interval_map = {}
    if col.dtype in [object, "category"] or col.nunique() <= bins:
        col_bin = col.astype(str)
        bin_interval_map = {str(v): str(v) for v in col_bin.dropna().unique()}
    else:
        if method == "quantile":
            _, raw_bins = pd.qcut(col, q=bins, duplicates="drop", labels=False, retbins=True)
            raw_bins = raw_bins.tolist()
            raw_bins[0] = -np.inf
            raw_bins[-1] = np.inf
            col_bin = pd.cut(col, bins=raw_bins, labels=False, include_lowest=True, duplicates="drop")
        elif method == "tree":
            thresholds = tree_bin_thresholds(col.dropna(), y[col.notna()], bins=bins)
            raw_bins = [-np.inf] + thresholds + [np.inf]
            col_bin = pd.cut(col, bins=raw_bins,
                             labels=False, duplicates="drop")
        else:
            _, raw_bins = pd.cut(col, bins=bins, labels=False, duplicates="drop", retbins=True)
            raw_bins = raw_bins.tolist()
            raw_bins[0] = -np.inf
            raw_bins[-1] = np.inf
            col_bin = pd.cut(col, bins=raw_bins, labels=False, include_lowest=True, duplicates="drop")
        for idx in range(len(raw_bins) - 1):
            left = raw_bins[idx]
            right = raw_bins[idx + 1]
            left_label = "-inf" if np.isneginf(left) else f"{left:.6f}"
            right_label = "inf" if np.isposinf(right) else f"{right:.6f}"
            bin_interval_map[idx] = f"({left_label}, {right_label}]"
        # missing values form their own bin (-1) instead of being dropped
        if col.isna().any():
            col_bin = pd.Series(col_bin, index=col.index)
            col_bin[col.isna()] = -1
            bin_interval_map[-1] = "MISSING"

    # WOE = ln(good_pct / bad_pct) — industry standard: higher WOE = lower risk
    grouped = woe_stats(col_bin, y)
    # IV = Σ(good_pct - bad_pct) * WOE  — non-negative when consistent with WOE direction
    grouped["iv_bin"] = (grouped["good_pct"] - grouped["bad_pct"]) * grouped["woe"]

    iv = grouped["iv_bin"].sum()
    grouped["iv"] = iv
    grouped["feature"] = feature
    grouped["bin_interval"] = grouped.index.map(lambda x: bin_interval_map.get(x, str(x)))

    return grouped.reset_index()


def classify_iv(iv: float) -> str:
    if iv < 0.02:   return "无预测力"
    if iv < 0.1:    return "弱"
    if iv < 0.3:    return "中"
    if iv < 0.5:    return "强"
    return "极强（需检查是否泄漏）"


# ============================================================
# 单调性检验
# ============================================================
def check_monotonicity(woe_df: pd.DataFrame, min_spearman: float = 0.6) -> dict:
    """检验 WOE 是否单调，返回 spearman 相关系数和是否单调"""
    woe_vals = woe_df["woe"].values
    bins_idx = np.arange(len(woe_vals))
    if len(woe_vals) < 3:
        return {"spearman": np.nan, "is_monotone": False}
    corr, pval = stats.spearmanr(bins_idx, woe_vals)
    is_monotone = abs(corr) >= min_spearman
    return {"spearman": round(corr, 4), "is_monotone": is_monotone, "pval": round(pval, 4)}


# ============================================================
# 主 EDA 类
# ============================================================
class AutoEDA:
    def __init__(self, config: dict = None):
        from config.config import EDA_CONFIG
        self.cfg = config or EDA_CONFIG
        self.results = {}

    # ----------------------------------------------------------
    def run(self, df: pd.DataFrame, label_col: str,
            report_dir: str = "reports/eda", time_col: str = None) -> dict:
        """
        一键运行全量 EDA，返回分析结果字典。
        """
        os.makedirs(report_dir, exist_ok=True)
        logger.info("=" * 50)
        logger.info("开始 Auto-EDA 分析")
        logger.info("=" * 50)

        feature_cols = [c for c in df.columns if c != label_col and c != time_col]

        # 1. 数据质量
        quality = self._data_quality(df, label_col)

        # 2. IV / WOE（仅对数值型特征）
        iv_table, woe_tables = self._iv_analysis(df, label_col, feature_cols)

        # 3. 单调性检验
        monotone_res = self._monotonicity(woe_tables)

        # 4. 相关性分析
        corr_matrix, high_corr_pairs = self._correlation(df, feature_cols)

        # 5. VIF
        vif_table = self._vif(df, feature_cols)

        # 6. PSI（有时间列时按时间前后二分，否则随机二分）
        psi_table = self._psi_internal(df, label_col, feature_cols, time_col=time_col)

        # 7. 综合评分
        summary = self._feature_summary(
            iv_table, monotone_res, psi_table, high_corr_pairs, vif_table
        )

        # 8. 可视化
        self._plot_iv_bar(iv_table, report_dir)
        self._plot_corr_heatmap(corr_matrix, report_dir)
        self._plot_woe_trends(woe_tables, monotone_res, report_dir)
        woe_bin_report = self._save_woe_bin_report(woe_tables, iv_table, report_dir)

        self.results = {
            "quality": quality,
            "iv_table": iv_table,
            "woe_tables": woe_tables,
            "woe_bin_report": woe_bin_report,
            "monotone": monotone_res,
            "corr_matrix": corr_matrix,
            "high_corr_pairs": high_corr_pairs,
            "vif_table": vif_table,
            "psi_table": psi_table,
            "summary": summary,
        }

        # 9. 输出汇总报告
        self._print_summary(summary)
        logger.info(f"EDA 完成，图表已保存至 {report_dir}")
        return self.results

    # ----------------------------------------------------------
    def _data_quality(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        logger.info("► 数据质量检测...")
        rows = []
        for col in df.columns:
            missing_cnt = df[col].isna().sum()
            missing_pct = missing_cnt / len(df)
            unique_cnt = df[col].nunique()
            dtype = str(df[col].dtype)
            rows.append({
                "feature": col,
                "dtype": dtype,
                "missing_count": missing_cnt,
                "missing_pct": round(missing_pct, 4),
                "unique_count": unique_cnt,
                "is_high_missing": missing_pct > self.cfg["missing_threshold"],
            })
        qdf = pd.DataFrame(rows)
        high_missing = qdf[qdf["is_high_missing"]]["feature"].tolist()
        if high_missing:
            logger.warning(f"高缺失率变量（>{self.cfg['missing_threshold']*100}%）: {high_missing}")
        return qdf

    # ----------------------------------------------------------
    def _iv_analysis(self, df, label_col, feature_cols):
        logger.info("► 计算 IV / WOE ...")
        iv_rows = []
        woe_tables = {}
        # Include both numeric and categorical features (nunique > 1)
        eligible_cols = [c for c in feature_cols if df[c].nunique() > 1 and df[c].nunique() < len(df) * 0.5]

        woe_bins = self.cfg.get("woe_bins", 10)
        for col in eligible_cols:
            try:
                # keep rows with missing feature values: they get their own bin
                woe_df = calc_woe_iv(df[[col, label_col]].dropna(subset=[label_col]),
                                     col, label_col, bins=woe_bins, method="tree")
                iv = woe_df["iv"].iloc[0]
                woe_tables[col] = woe_df
                iv_rows.append({
                    "feature": col,
                    "IV": round(iv, 4),
                    "IV_label": classify_iv(iv),
                })
            except Exception as e:
                logger.warning(f"  {col} IV 计算失败: {e}")

        iv_table = pd.DataFrame(iv_rows).sort_values("IV", ascending=False)
        return iv_table, woe_tables

    # ----------------------------------------------------------
    def _save_woe_bin_report(self, woe_tables: dict, iv_table: pd.DataFrame, report_dir: str) -> pd.DataFrame:
        if not woe_tables:
            out = pd.DataFrame()
            out.to_csv(f"{report_dir}/woe_bin_report.csv", index=False)
            return out
        iv_rank = dict(zip(iv_table["feature"], range(1, len(iv_table) + 1))) if len(iv_table) else {}
        frames = []
        for feature, woe_df in woe_tables.items():
            tmp = woe_df.copy()
            tmp["feature"] = feature
            tmp["iv_rank"] = iv_rank.get(feature, np.nan)
            tmp["bin_label"] = tmp.get("bin_interval", tmp["bin"].astype(str))
            tmp["bad_rate"] = tmp["bad_rate"].round(6)
            tmp["woe"] = tmp["woe"].round(6)
            tmp["iv_bin"] = tmp["iv_bin"].round(6)
            tmp["iv"] = tmp["iv"].round(6)
            frames.append(tmp)
        out = pd.concat(frames, ignore_index=True)
        cols = [
            "feature",
            "iv_rank",
            "bin",
            "bin_label",
            "bin_interval",
            "total",
            "bad",
            "good",
            "bad_rate",
            "woe",
            "iv_bin",
            "iv",
            "bad_pct",
            "good_pct",
        ]
        cols = [c for c in cols if c in out.columns]
        out = out[cols].sort_values(["iv_rank", "bin_label"]).reset_index(drop=True)
        out.to_csv(f"{report_dir}/woe_bin_report.csv", index=False)
        return out

    # ----------------------------------------------------------
    def _monotonicity(self, woe_tables: dict) -> dict:
        logger.info("► 单调性检验 ...")
        min_spearman = self.cfg.get("monotone_spearman_min", 0.6)
        res = {}
        for feat, woe_df in woe_tables.items():
            # the missing bin (-1) has no natural order; exclude it from the trend check
            ordered = woe_df[woe_df["bin"] != -1] if "bin" in woe_df.columns else woe_df
            res[feat] = check_monotonicity(ordered, min_spearman=min_spearman)
        return res

    # ----------------------------------------------------------
    def _correlation(self, df, feature_cols):
        logger.info("► 相关性分析 ...")
        num_cols = [c for c in feature_cols
                    if df[c].dtype in [np.float64, np.int64, float, int]]
        corr = df[num_cols].corr(method="spearman")
        threshold = self.cfg["corr_threshold"]

        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if val >= threshold:
                    high_corr_pairs.append({
                        "feature_a": corr.columns[i],
                        "feature_b": corr.columns[j],
                        "correlation": round(corr.iloc[i, j], 4),
                    })
        if high_corr_pairs:
            logger.warning(f"发现 {len(high_corr_pairs)} 对高相关变量对（阈值={threshold}）")
        return corr, high_corr_pairs

    # ----------------------------------------------------------
    def _vif(self, df, feature_cols) -> pd.DataFrame:
        logger.info("► VIF 计算 ...")
        num_cols = [c for c in feature_cols
                    if df[c].dtype in [np.float64, np.int64, float, int]]
        sub = df[num_cols].dropna()
        if sub.shape[1] < 2:
            return pd.DataFrame()
        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = sub.columns
            vif_data["VIF"] = [variance_inflation_factor(sub.values, i)
                               for i in range(sub.shape[1])]
            vif_data["high_vif"] = vif_data["VIF"] > self.cfg["vif_threshold"]
            return vif_data.sort_values("VIF", ascending=False)
        except Exception as e:
            logger.warning(f"VIF 计算失败: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    def _psi_internal(self, df, label_col, feature_cols, time_col: str = None) -> pd.DataFrame:
        """特征 PSI：有时间列时按时间前后二分（检测真实漂移），否则随机二分兜底"""
        logger.info("► PSI 稳定性计算 ...")
        if time_col and time_col in df.columns:
            times = pd.to_datetime(df[time_col], errors="coerce")
            df_ordered = df.loc[times.sort_values(kind="stable").index]
            logger.info(f"  按时间列 '{time_col}' 前后二分计算 PSI")
        else:
            df_ordered = df.sample(frac=1.0, random_state=42)
            logger.info("  无时间列，随机二分计算 PSI（仅作参考，检测不到时序漂移）")
        mid = len(df_ordered) // 2
        df1, df2 = df_ordered.iloc[:mid], df_ordered.iloc[mid:]
        psi_stable = self.cfg.get("psi_stable", 0.1)
        psi_warning = self.cfg.get("psi_warning", 0.25)
        rows = []
        num_cols = [c for c in feature_cols
                    if df[c].dtype in [np.float64, np.int64, float, int]]
        for col in num_cols:
            try:
                psi_val = calc_psi(df1[col].dropna().values, df2[col].dropna().values)
                label = ("稳定" if psi_val < psi_stable
                         else "警告" if psi_val < psi_warning
                         else "不稳定")
                rows.append({"feature": col, "PSI": round(psi_val, 4), "PSI_label": label})
            except Exception:
                pass
        return pd.DataFrame(rows).sort_values("PSI", ascending=False)

    # ----------------------------------------------------------
    def _feature_summary(self, iv_table, monotone_res, psi_table,
                          high_corr_pairs, vif_table) -> pd.DataFrame:
        """综合所有维度，输出特征评分汇总表"""
        high_corr_feats = set()
        iv_map = dict(zip(iv_table["feature"], iv_table["IV"])) if len(iv_table) else {}

        for pair in high_corr_pairs:
            a, b = pair["feature_a"], pair["feature_b"]
            # 保留 IV 更高的变量
            drop = b if iv_map.get(a, 0) >= iv_map.get(b, 0) else a
            high_corr_feats.add(drop)

        high_vif_feats = set()
        if len(vif_table):
            high_vif_feats = set(vif_table[vif_table["high_vif"]]["feature"].tolist())

        psi_map = dict(zip(psi_table["feature"], psi_table["PSI"])) if len(psi_table) else {}

        rows = []
        for feat, iv_val in iv_map.items():
            mono = monotone_res.get(feat, {})
            psi_val = psi_map.get(feat, np.nan)
            recommend = "OK"
            issues = []
            if feat in high_corr_feats:
                issues.append("高相关待剔除")
            if feat in high_vif_feats:
                issues.append("高VIF")
            if not mono.get("is_monotone", True):
                issues.append("WOE不单调")
            if psi_val > self.cfg.get("psi_stable", 0.1):
                issues.append(f"PSI={psi_val:.3f}不稳定")
            if iv_val < 0.02:
                issues.append("IV过低")
            if issues:
                recommend = "WARN:" + "，".join(issues)

            rows.append({
                "feature": feat,
                "IV": iv_val,
                "IV_label": classify_iv(iv_val),
                "WOE_Spearman": mono.get("spearman", np.nan),
                "is_monotone": mono.get("is_monotone", False),
                "PSI": psi_val,
                "high_corr": feat in high_corr_feats,
                "high_vif": feat in high_vif_feats,
                "recommendation": recommend,
            })

        return pd.DataFrame(rows).sort_values("IV", ascending=False)

    # ----------------------------------------------------------
    def _plot_iv_bar(self, iv_table, report_dir):
        if len(iv_table) == 0:
            return
        top = iv_table.head(30)
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
        colors = ["#2ecc71" if iv >= 0.1 else "#e67e22" if iv >= 0.02 else "#e74c3c"
                  for iv in top["IV"]]
        ax.barh(top["feature"][::-1], top["IV"][::-1], color=colors[::-1])
        ax.axvline(0.02, color="orange", linestyle="--", linewidth=1, label="弱(0.02)")
        ax.axvline(0.1, color="green", linestyle="--", linewidth=1, label="中(0.10)")
        ax.set_xlabel("IV")
        ax.set_title("Feature IV Ranking (Top 30)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{report_dir}/iv_ranking.png", dpi=120)
        plt.close()

    def _plot_corr_heatmap(self, corr_matrix, report_dir):
        if corr_matrix is None or corr_matrix.shape[0] == 0:
            return
        n = min(corr_matrix.shape[0], 30)
        sub = corr_matrix.iloc[:n, :n]
        fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.45)))
        mask = np.triu(np.ones_like(sub, dtype=bool))
        sns.heatmap(sub, mask=mask, cmap="RdYlGn", center=0,
                    annot=(n <= 15), fmt=".2f", ax=ax, linewidths=0.3)
        ax.set_title("Feature Correlation Matrix (Spearman)")
        plt.tight_layout()
        plt.savefig(f"{report_dir}/corr_heatmap.png", dpi=120)
        plt.close()

    def _plot_woe_trends(self, woe_tables, monotone_res, report_dir):
        feats = list(woe_tables.keys())[:16]
        if not feats:
            return
        cols_n = 4
        rows_n = (len(feats) + cols_n - 1) // cols_n
        fig, axes = plt.subplots(rows_n, cols_n,
                                 figsize=(cols_n * 4, rows_n * 3))
        axes = np.array(axes).flatten()
        for i, feat in enumerate(feats):
            ax = axes[i]
            wdf = woe_tables[feat]
            ax.plot(range(len(wdf)), wdf["woe"], marker="o",
                    color="#3498db", linewidth=1.5)
            mono = monotone_res.get(feat, {})
            color = "#2ecc71" if mono.get("is_monotone") else "#e74c3c"
            ax.set_title(f"{feat}\nSpearman={mono.get('spearman','N/A')}",
                         fontsize=8, color=color)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.tick_params(labelsize=6)
        for j in range(len(feats), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("WOE Trend by Feature (Green=Monotone)", y=1.01, fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{report_dir}/woe_trends.png", dpi=120, bbox_inches="tight")
        plt.close()

    def _print_summary(self, summary: pd.DataFrame):
        logger.info("\n" + "=" * 60)
        logger.info("特征综合评估汇总（Top 20）")
        logger.info("=" * 60)
        cols = ["feature", "IV", "IV_label", "is_monotone", "PSI",
                "high_corr", "recommendation"]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].head(20).to_string(index=False))
        logger.info("=" * 60)
