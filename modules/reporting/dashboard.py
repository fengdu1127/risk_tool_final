"""Generate a static visual dashboard from pipeline CSV/PNG reports."""
import html
import json
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _table(df: pd.DataFrame, columns: list[str] = None, max_rows: int = 8) -> str:
    if df.empty:
        return '<p class="muted">暂无数据</p>'
    if columns:
        available = [c for c in columns if c in df.columns]
        if not available:
            return '<p class="muted">暂无数据</p>'
        view = df[available]
    else:
        view = df
    view = view.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    body = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in view.columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _metric_card(title: str, value: str, subtitle: str = "", level: str = "") -> str:
    cls = f"metric {level}".strip()
    return (
        f'<section class="{cls}"><span>{html.escape(title)}</span>'
        f"<strong>{html.escape(value)}</strong><small>{html.escape(subtitle)}</small></section>"
    )


def _bar_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str, max_rows: int = 8) -> str:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return '<p class="muted">暂无数据</p>'
    chart_df = df[[label_col, value_col]].dropna().head(max_rows)
    if chart_df.empty:
        return '<p class="muted">暂无数据</p>'
    max_val = max(float(chart_df[value_col].max()), 1e-9)
    rows = []
    for _, row in chart_df.iterrows():
        value = float(row[value_col])
        width = min(100, max(2, value / max_val * 100))
        rows.append(
            '<div class="bar-row">'
            f'<span>{html.escape(str(row[label_col]))}</span>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.2f}%"></div>'
            "</div>"
            f"<b>{value:.4f}</b>"
            "</div>"
        )
    return f"<h3>{html.escape(title)}</h3><div class=\"bars\">{''.join(rows)}</div>"


def _image(path: Path, title: str, root: Path = None) -> str:
    check_path = (root / path) if root is not None else path
    if not check_path.exists():
        return ""
    rel = path.as_posix()
    return (
        '<figure class="figure">'
        f'<img src="{html.escape(rel)}" alt="{html.escape(title)}">'
        f"<figcaption>{html.escape(title)}</figcaption>"
        "</figure>"
    )


def _risk_notes(model_metrics: pd.DataFrame, overfit: pd.DataFrame, stability: pd.DataFrame) -> list[str]:
    notes = []
    if not overfit.empty:
        train_test_gap = float(overfit.iloc[0].get("train_test_ks_gap", 0) or 0)
        test_valid_gap = float(overfit.iloc[0].get("test_valid_ks_gap", 0) or 0)
        if train_test_gap > 0.10:
            notes.append(f"模型存在过拟合风险：train-test KS gap={train_test_gap:.4f}，超过 10% 目标。")
        else:
            notes.append(f"模型 train-test KS gap={train_test_gap:.4f}，满足 10% 以内目标。")
        if abs(test_valid_gap) <= 0.05:
            notes.append(f"最终验证稳定性较好：test-valid KS gap={test_valid_gap:.4f}。")
    if {"feature", "dataset"}.issubset(stability.columns):
        score_psi = stability[(stability["feature"] == "model_score") & (stability["dataset"] == "valid")]
    else:
        score_psi = pd.DataFrame()
    if not score_psi.empty:
        psi = float(score_psi.iloc[0].get("PSI", 0) or 0)
        level = "稳定" if psi < 0.1 else "需关注"
        notes.append(f"valid 分数 PSI={psi:.4f}，当前判断为{level}。")
    if model_metrics.empty:
        notes.append("模型指标缺失，请检查 model_comparison.csv 是否生成。")
    return notes


def generate_dashboard(output_dir: str, html_path: str = None) -> str:
    root = Path(output_dir)
    eda_dir = root / "eda"
    model_dir = root / "model"
    strategy_dir = root / "strategy"
    html_path = html_path or str(root / "dashboard.html")

    summary = _read_json(root / "run_summary.json")
    split_profile = _read_csv(root / "split_profile.csv")
    woe_bins = _read_csv(eda_dir / "woe_bin_report.csv")
    model_metrics = _read_csv(model_dir / "model_comparison.csv")
    overfit = _read_csv(model_dir / "overfit_diagnostics.csv")
    score_bins = _read_csv(model_dir / "score_bins.csv")
    score_bin_stability = _read_csv(model_dir / "score_bin_stability.csv")
    stability = _read_csv(model_dir / "stability_report.csv")
    leaderboard = _read_csv(strategy_dir / "strategy_leaderboard.csv")
    recommendation = _read_csv(strategy_dir / "strategy_recommendation.csv")
    segment_disc = _read_csv(strategy_dir / "segment_discrimination.csv")
    segment_candidates = _read_csv(strategy_dir / "segment_strategy_candidates.csv")
    segment_recommendation = _read_csv(strategy_dir / "segment_strategy_recommendation.csv")
    final_comparison = _read_csv(strategy_dir / "final_strategy_comparison.csv")
    rule_stability = _read_csv(strategy_dir / "rule_stability.csv")

    valid_metrics = model_metrics[model_metrics.get("dataset", pd.Series(dtype=str)) == "valid"]
    best_valid = valid_metrics.iloc[0] if not valid_metrics.empty else {}
    diagnostics = overfit.iloc[0] if not overfit.empty else {}
    cards = [
        _metric_card("样本量", str(summary.get("n_total", "-")), "总样本"),
        _metric_card("Valid KS", _fmt(best_valid.get("KS", None)), "最终验证集"),
        _metric_card(
            "Train-Test KS Gap",
            _fmt(diagnostics.get("train_test_ks_gap", None)),
            "目标 <= 0.1000",
            "" if diagnostics.get("train_test_ks_gap", 1) <= 0.10 else "warn",
        ),
        _metric_card("Test-Valid KS Gap", _fmt(diagnostics.get("test_valid_ks_gap", None)), "泛化稳定性"),
    ]
    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in _risk_notes(model_metrics, overfit, stability))

    score_bin_view = score_bins.copy()
    if not score_bin_view.empty:
        score_bin_view["bin_label"] = score_bin_view.get("score_bin_interval", score_bin_view.get("score_bin", pd.Series(dtype=str))).astype(str)
    valid_bins = score_bin_view[score_bin_view.get("dataset", pd.Series(dtype=str)) == "valid"].copy()
    if {"dataset", "PSI"}.issubset(stability.columns):
        psi_valid = stability[stability["dataset"] == "valid"].sort_values("PSI", ascending=False)
    else:
        psi_valid = pd.DataFrame()
    segment_view = segment_disc.copy()
    if not segment_view.empty:
        segment_view["status"] = segment_view["recommend_as_segment_feature"].map({True: "通过", False: "不通过"})
    top_woe_bins = woe_bins.copy()
    if not top_woe_bins.empty:
        top_features = (
            top_woe_bins[["feature", "iv"]]
            .drop_duplicates()
            .sort_values("iv", ascending=False)
            .head(5)["feature"]
            .tolist()
        )
        top_woe_bins = top_woe_bins[top_woe_bins["feature"].isin(top_features)].copy()
        top_woe_bins["feature_bin"] = top_woe_bins["feature"].astype(str) + " | " + top_woe_bins["bin_label"].astype(str)
    recommended_segment = segment_recommendation[
        (segment_recommendation.get("dataset", pd.Series(dtype=str)) == "valid")
        & (segment_recommendation.get("action", pd.Series(dtype=str)) == "reject")
    ].copy()
    segment_candidate_view = segment_candidates[
        (segment_candidates.get("dataset", pd.Series(dtype=str)) == "valid")
        & (segment_candidates.get("action", pd.Series(dtype=str)) == "reject")
    ].copy()
    final_valid = final_comparison[final_comparison.get("dataset", pd.Series(dtype=str)) == "valid"].copy()
    woe_bad_rate_view = (
        top_woe_bins.sort_values("bad_rate", ascending=False)
        if {"feature_bin", "bad_rate"}.issubset(top_woe_bins.columns)
        else pd.DataFrame()
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Risk Tool 可视化报告</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f6f7f9; color: #18212f; }}
    header {{ padding: 28px 40px 20px; background: #ffffff; border-bottom: 1px solid #dfe4ea; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    h3 {{ margin: 0 0 12px; font-size: 15px; }}
    main {{ padding: 24px 40px 40px; display: grid; gap: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .two {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
    .panel, .metric {{ background: #fff; border: 1px solid #dfe4ea; border-radius: 8px; padding: 18px; }}
    .metric span {{ display: block; font-size: 13px; color: #5a6678; }}
    .metric strong {{ display: block; margin: 8px 0 6px; font-size: 26px; }}
    .metric small, .muted {{ color: #6b7788; }}
    .warn strong {{ color: #b45309; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid #e8edf2; text-align: left; white-space: nowrap; }}
    th {{ color: #44546a; background: #f7f9fb; }}
    .bars {{ display: grid; gap: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: 110px 1fr 70px; align-items: center; gap: 10px; font-size: 13px; }}
    .bar-track {{ height: 16px; background: #e9eef4; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: #2563eb; }}
    .figure img {{ width: 100%; max-height: 360px; object-fit: contain; background: #fff; }}
    .figure figcaption {{ margin-top: 8px; color: #5a6678; font-size: 13px; }}
    ul {{ margin: 0; padding-left: 20px; line-height: 1.8; }}
    @media (max-width: 980px) {{ .grid, .two {{ grid-template-columns: 1fr; }} header, main {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Risk Tool 可视化报告</h1>
    <p class="muted">输出目录：{html.escape(str(root))} ｜ 模型：{html.escape(str(summary.get("best_model", "-")))} ｜ Valid 月份：{html.escape(str(summary.get("valid_months", "-")))}</p>
  </header>
  <main>
    <section class="grid">{''.join(cards)}</section>
    <section class="panel"><h2>当前主要问题</h2><ul>{notes_html}</ul></section>
    <section class="two">
      <div class="panel"><h2>数据切分画像</h2>{_table(split_profile, max_rows=5)}</div>
      <div class="panel"><h2>模型指标</h2>{_table(model_metrics, ["model", "dataset", "AUC", "KS", "Gini"], 8)}</div>
    </section>
    <section class="two">
      <div class="panel"><h2>模型评分切档效果（Train/Test/Valid）</h2>{_table(score_bin_view, ["dataset", "bin_order", "score_bin_interval", "count", "bad_count", "raw_bad_rate", "monotone_bad_rate", "lift", "monotone_lift", "cum_bad_capture", "bad_rate_gap_vs_train", "monotone_bad_rate_gap_vs_train"], 36)}</div>
      <div class="panel">{_bar_chart(psi_valid, "feature", "PSI", "Valid PSI Top")}</div>
    </section>
    <section class="two">
      <div class="panel"><h2>评分十分箱单调稳定性</h2>{_table(score_bin_stability, ["dataset", "n_bins", "raw_bad_rate_monotone", "monotone_bad_rate_monotone", "max_abs_bad_rate_gap_vs_train", "max_abs_monotone_bad_rate_gap_vs_train"], 6)}</div>
      <div class="panel">{_bar_chart(valid_bins, "bin_label", "lift", "Valid 十分箱 Lift", 10)}</div>
    </section>
    <section class="two">
      <div class="panel"><h2>单变量分箱风险表现（Top IV）</h2>{_table(top_woe_bins, ["feature", "bin_label", "total", "bad", "bad_rate", "woe", "iv_bin", "iv"], 30)}</div>
      <div class="panel">{_bar_chart(woe_bad_rate_view, "feature_bin", "bad_rate", "分箱坏账率 Top", 12)}</div>
    </section>
    <section class="two">
      <div class="panel"><h2>策略 Leaderboard</h2>{_table(leaderboard, ["strategy_name", "valid_reject_rate", "valid_reject_bad_rate", "valid_lift", "test_valid_lift_gap", "leaderboard_score"], 6)}</div>
      <div class="panel"><h2>三档策略推荐</h2>{_table(recommendation, ["recommendation_type", "strategy_name", "valid_reject_rate", "valid_lift", "leaderboard_score"], 6)}</div>
    </section>
    <section class="panel"><h2>最终策略明细（Valid）</h2>{_table(final_valid, ["strategy_name", "reject_rate", "review_rate", "approve_rate", "reject_bad_rate", "approve_bad_rate", "lift"], 10)}</section>
    <section class="two">
      <div class="panel"><h2>分客群策略推荐明细（Valid Reject）</h2>{_table(recommended_segment, ["segment_feature", "segment_value", "reject_threshold", "count", "rate", "bad_rate", "lift", "recommend_segment_strategy", "abs_lift_gap_test_valid", "fallback_to_global"], 12)}</div>
      <div class="panel"><h2>分客群候选阈值（Valid Reject）</h2>{_table(segment_candidate_view, ["policy_id", "segment_feature", "segment_value", "reject_threshold", "count", "rate", "bad_rate", "lift"], 12)}</div>
    </section>
    <section class="two">
      <div class="panel"><h2>客群区分度筛选</h2>{_table(segment_view, ["segment_feature", "train_share_gap", "valid_bad_rate_gap", "stable_order", "status"], 8)}</div>
      <div class="panel"><h2>稳定规则</h2>{_table(rule_stability, ["rule_id", "condition", "lift_train", "lift_test", "lift_valid", "stable_pass"], 8)}</div>
    </section>
    <section class="two">
      <div class="panel">{_image(Path("model") / "xgb_roc_ks_valid.png", "Valid ROC / KS", root)}</div>
      <div class="panel">{_image(Path("model") / "xgb_lift_valid.png", "Valid Lift", root)}</div>
    </section>
    <section class="two">
      <div class="panel">{_image(Path("model") / "xgb_feat_importance.png", "特征重要性", root)}</div>
      <div class="panel">{_image(Path("strategy") / "rule_comparison.png", "规则稳定性对比", root)}</div>
    </section>
  </main>
</body>
</html>
"""
    Path(html_path).write_text(html_text, encoding="utf-8")
    return str(Path(html_path))
