"""A single self-contained HTML report per run.

Images are embedded as data URIs and the styling is inline, so the file can be
attached to an email or handed to a model-risk reviewer without a web server,
a network connection or any of the run's other files.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from ..logging_setup import get_logger
from .plots import make_figures

log = get_logger("report")

_STYLE = """
:root { color-scheme: light dark; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --bg:#ffffff; --accent:#2f5d8a; --chip:#f4f6f8; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --muted:#a0a0a0; --line:#333; --bg:#161616; --accent:#7fb0dd; --chip:#242424; }
}
* { box-sizing: border-box; }
body { margin:0 auto; padding:2rem 1.5rem 4rem; max-width:1080px; background:var(--bg); color:var(--fg);
       font:15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size:1.6rem; margin:0 0 .25rem; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem; border-bottom:1px solid var(--line); }
.sub { color:var(--muted); margin:0 0 1.5rem; }
.cards { display:flex; flex-wrap:wrap; gap:.75rem; margin:1rem 0 0; }
.card { flex:1 1 150px; background:var(--chip); border:1px solid var(--line); border-radius:8px; padding:.7rem .85rem; }
.card .k { display:block; font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.card .v { display:block; font-size:1.25rem; font-weight:600; margin-top:.15rem; color:var(--accent); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.86rem; margin:.5rem 0; }
th, td { text-align:right; padding:.4rem .6rem; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.03em; }
tbody tr:hover { background:var(--chip); }
img { max-width:100%; height:auto; display:block; margin:.75rem 0; }
ul { padding-left:1.2rem; }
li { margin:.2rem 0; }
code { background:var(--chip); padding:.1rem .3rem; border-radius:3px; font-size:.85em; }
.note { color:var(--muted); font-size:.85rem; margin:.4rem 0 0; }
"""


def write_report(run, make_plots: bool = True, dpi: int = 120) -> Path:
    figures = make_figures(run, dpi) if make_plots else []
    summary = run.summary()
    parts: list[str] = [
        f"<h1>Risk model run <code>{html.escape(run.name)}</code></h1>",
        f"<p class='sub'>{_headline(summary)}</p>",
        _cards(summary),
        _model_section(run, summary, figures),
        _features_section(run, figures),
        _policy_section(run, summary, figures),
        _outcome_section(run, summary),
    ]
    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>riskflow — {html.escape(run.name)}</title><style>{_STYLE}</style></head>"
        f"<body>{''.join(p for p in parts if p)}</body></html>"
    )
    run.report_path.write_text(document, encoding="utf-8")
    log.info("wrote report to %s", run.report_path)
    return run.report_path


def _headline(summary: dict) -> str:
    model = summary.get("model", {})
    split = summary.get("split", {})
    rows = split.get("rows", {})
    return html.escape(
        f"{model.get('algorithm', 'model')} via {model.get('backend', 'unknown backend')} · "
        f"{split.get('strategy', 'split')} · "
        f"{rows.get('train', 0):,} train / {rows.get('test', 0):,} test / {rows.get('holdout', 0):,} holdout"
    )


def _cards(summary: dict) -> str:
    metrics = {m["dataset"]: m for m in summary.get("model", {}).get("metrics", []) if m.get("model") == summary.get("model", {}).get("algorithm")}
    outcome = summary.get("outcome_on_holdout", {})
    cards = [
        ("Holdout KS", _fmt(metrics.get("holdout", {}).get("ks"), 3)),
        ("Holdout AUC", _fmt(metrics.get("holdout", {}).get("auc"), 3)),
        ("Overfit", summary.get("model", {}).get("diagnostics", {}).get("overfit_verdict", "—")),
        ("Decline rate", _fmt(outcome.get("reject_rate"), 1, percent=True)),
        ("Bads caught", _fmt(outcome.get("bad_capture_at_reject"), 1, percent=True)),
        ("Approved bad rate", _fmt(outcome.get("approved_bad_rate"), 2, percent=True)),
    ]
    chips = "".join(
        f"<div class='card'><span class='k'>{html.escape(k)}</span><span class='v'>{html.escape(str(v))}</span></div>"
        for k, v in cards
    )
    return f"<div class='cards'>{chips}</div>"


def _model_section(run, summary: dict, figures) -> str:
    blocks = ["<h2>Model</h2>"]
    metrics = run.table("model_metrics")
    if metrics is not None:
        blocks.append(_table(metrics))
    diagnostics = summary.get("model", {}).get("diagnostics", {})
    if diagnostics:
        items = "".join(f"<li>{html.escape(k.replace('_', ' '))}: <code>{html.escape(str(v))}</code></li>" for k, v in diagnostics.items())
        blocks.append(f"<ul>{items}</ul>")
    blocks.append(_image(figures, "gains_by_band"))
    blocks.append(_image(figures, "calibration"))
    blocks.append(
        "<p class='note'>Calibration is fitted on out-of-fold predictions over the training sample, so the "
        "test set stays reserved for choosing between models.</p>"
    )
    return "".join(blocks)


def _features_section(run, figures) -> str:
    blocks = ["<h2>Features</h2>", _image(figures, "information_value"), _image(figures, "woe_trends")]
    screening = run.table("screening")
    if screening is not None and len(screening):
        dropped = screening[~screening["selected"]]
        if len(dropped):
            blocks.append("<p class='note'>Features that did not make it, and why:</p>")
            blocks.append(_table(dropped[["feature", "iv", "reason"]], limit=20))
    return "".join(blocks)


def _policy_section(run, summary: dict, figures) -> str:
    blocks = ["<h2>Decision policy</h2>"]
    cutoff = summary.get("policy", {}).get("global_cutoff", {})
    if cutoff:
        blocks.append(
            f"<p>Decline at a model score of <code>{cutoff.get('reject_at', 0):.4f}</code> or above; "
            f"refer for manual review from <code>{cutoff.get('review_at', 0):.4f}</code>.</p>"
        )
    overrides = summary.get("policy", {}).get("segment_overrides", [])
    if overrides:
        items = "".join(
            f"<li><code>{html.escape(str(o['feature']))} = {html.escape(str(o['value']))}</code> "
            f"declines from <code>{o['cutoff']['reject_at']:.4f}</code></li>"
            for o in overrides
        )
        blocks.append(f"<p>Segment overrides:</p><ul>{items}</ul>")
    rules = summary.get("rules", {})
    if rules.get("descriptions"):
        items = "".join(f"<li><code>{html.escape(d)}</code></li>" for d in rules["descriptions"])
        blocks.append(
            f"<p>{rules['stable']} of {rules['mined']} mined rules held up across train, test and holdout:</p><ul>{items}</ul>"
        )
    elif rules.get("mined"):
        blocks.append(f"<p class='note'>{rules['mined']} rules were mined but none survived out-of-sample validation.</p>")
    blocks.append(_image(figures, "cutoff_tradeoff"))
    return "".join(blocks)


def _outcome_section(run, summary: dict) -> str:
    decisions = run.table("decisions")
    if decisions is None or decisions.empty:
        return ""
    return "<h2>What the policy does</h2>" + _table(decisions)


def _table(frame: pd.DataFrame, limit: int = 40) -> str:
    if frame is None or frame.empty:
        return ""
    view = frame.head(limit)
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_cell(v))}</td>" for v in row) + "</tr>"
        for row in view.itertuples(index=False)
    )
    more = f"<p class='note'>Showing {limit} of {len(frame)} rows.</p>" if len(frame) > limit else ""
    return f"<div class='scroll'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>{more}"


def _cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:,.1f}"
    return str(value)


def _image(figures, stem: str) -> str:
    for path in figures:
        if Path(path).stem == stem:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            return f"<img alt='{html.escape(stem.replace('_', ' '))}' src='data:image/png;base64,{encoded}'>"
    return ""


def _fmt(value, digits: int, percent: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number * 100:.{digits}f}%" if percent else f"{number:.{digits}f}"
