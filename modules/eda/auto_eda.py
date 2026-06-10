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
from utils.helpers import get_logger, calc_psi

warnings.filterwarnings("ignore")
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
            from sklearn.tree import DecisionTreeClassifier
            dt = DecisionTreeClassifier(max_leaf_nodes=bins, min_samples_leaf=50)
            dt.fit(col.fillna(col.median()).values.reshape(-1, 1), y)
            thresholds = sorted(set(dt.tree_.threshold[dt.tree_.threshold != -2]))
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

    tmp = pd.DataFrame({"bin": col_bin, "label": y}).dropna(subset=["bin"])
    grouped = tmp.groupby("bin")["label"].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    grouped["bad_pct"] = grouped["bad"] / total_bad
    grouped["good_pct"] = grouped["good"] / total_good

    grouped["bad_pct"] = grouped["bad_pct"].replace(0, 1e-6)
    grouped["good_pct"] = grouped["good_pct"].replace(0, 1e-6)

    # WOE = ln(good_pct / bad_pct) — industry standard: higher WOE = lower risk
    grouped["woe"] = np.log(grouped["good_pct"] / grouped["bad_pct"])
    # IV = Σ(good_pct - bad_pct) * WOE  — non-negative when consistent with WOE direction
    grouped["iv_bin"] = (grouped["good_pct"] - grouped["bad_pct"]) * grouped["woe"]

    iv = grouped["iv_bin"].sum()
    grouped["iv"] = iv
    grouped["feature"] = feature
    grouped["bad_rate"] = grouped["bad"] / grouped["total"]
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
def check_monotonicity(woe_df: pd.DataFrame) -> dict:
    """检验 WOE 是否单调，返回 spearman 相关系数和是否单调"""
    woe_vals = woe_df["woe"].values
    bins_idx = np.arange(len(woe_vals))
    if len(woe_vals) < 3:
        return {"spearman": np.nan, "is_monotone": False}
    corr, pval = stats.spearmanr(bins_idx, woe_vals)
    is_monotone = abs(corr) >= 0.6
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
            report_dir: str = "reports/eda") -> dict:
        """
        一键运行全量 EDA，返回分析结果字典。
        """
        os.makedirs(report_dir, exist_ok=True)
        logger.info("=" * 50)
        logger.info("开始 Auto-EDA 分析")
        logger.info("=" * 50)

        feature_cols = [c for c in df.columns if c != label_col]

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

        # 6. PSI（训练集内按时间或随机二分）
        psi_table = self._psi_internal(df, label_col, feature_cols)

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

        for col in eligible_cols:
            try:
                woe_df = calc_woe_iv(df[[col, label_col]].dropna(),
                                     col, label_col, bins=10, method="tree")
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
        res = {}
        for feat, woe_df in woe_tables.items():
            res[feat] = check_monotonicity(woe_df)
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
    def _psi_internal(self, df, label_col, feature_cols) -> pd.DataFrame:
        """随机二分数据计算特征 PSI 作为稳定性参考（避免时序偏差）"""
        logger.info("► PSI 稳定性计算 ...")
        df_shuffled = df.sample(frac=1.0, random_state=42)
        mid = len(df_shuffled) // 2
        df1, df2 = df_shuffled.iloc[:mid], df_shuffled.iloc[mid:]
        rows = []
        num_cols = [c for c in feature_cols
                    if df[c].dtype in [np.float64, np.int64, float, int]]
        for col in num_cols:
            try:
                psi_val = calc_psi(df1[col].dropna().values, df2[col].dropna().values)
                label = ("稳定" if psi_val < 0.1
                         else "警告" if psi_val < 0.25
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
            if psi_val > 0.1:
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
