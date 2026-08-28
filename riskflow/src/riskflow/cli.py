"""Command line interface.

    riskflow demo                       generate data and train a run end to end
    riskflow train  --data … --label …  train and write a run directory
    riskflow score  --data …            score a batch with the promoted run
    riskflow promote <run>              point PRODUCTION at a run
    riskflow runs                       list runs
    riskflow compare <a> <b>            diff two runs
    riskflow explain --data …           show why one applicant got their decision
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .logging_setup import configure, get_logger
from .settings import Settings

log = get_logger("cli")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure()
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        log.error("%s", exc)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="riskflow", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs", help="directory holding run outputs (default: runs)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train a model and decision policy")
    train.add_argument("--data", required=True, help="input CSV")
    train.add_argument("--label", required=True, help="binary 0/1 outcome column")
    train.add_argument("--features", nargs="+", default=None, help="feature columns (default: everything else)")
    train.add_argument("--time-col", default=None, help="application timestamp, enabling an out-of-time holdout")
    train.add_argument("--id-col", default=None, help="identifier column to carry into scored output")
    train.add_argument("--config", default=None, help="JSON file overriding settings")
    train.add_argument("--name", default=None, help="run name (default: run_<timestamp>)")
    train.add_argument("--promote", action="store_true", help="promote the run if it completes")
    train.set_defaults(handler=_train)

    score = subparsers.add_parser("score", help="score a batch")
    score.add_argument("--data", required=True, help="input CSV")
    score.add_argument("--run", default=None, help="run to score with (default: the promoted one)")
    score.add_argument("--output", default=None, help="where to write scored rows")
    score.add_argument("--config", default=None, help="JSON file overriding settings")
    score.set_defaults(handler=_score)

    promote = subparsers.add_parser("promote", help="make a run the production model")
    promote.add_argument("run", help="run name or path")
    promote.set_defaults(handler=_promote)

    runs = subparsers.add_parser("runs", help="list runs")
    runs.set_defaults(handler=_runs)

    compare = subparsers.add_parser("compare", help="diff two runs")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--sample", default=None, help="CSV to measure decision flips on")
    compare.set_defaults(handler=_compare)

    explain = subparsers.add_parser("explain", help="explain the decision for one row")
    explain.add_argument("--data", required=True, help="CSV containing the applicant")
    explain.add_argument("--row", type=int, default=0, help="row index to explain (default: 0)")
    explain.add_argument("--run", default=None, help="run to use (default: the promoted one)")
    explain.set_defaults(handler=_explain)

    demo = subparsers.add_parser("demo", help="generate synthetic data and train on it")
    demo.add_argument("--rows", type=int, default=12000)
    demo.add_argument("--out", default="demo_data.csv", help="where to write the generated data")
    demo.add_argument("--name", default=None, help="run name")
    demo.set_defaults(handler=_demo)

    return parser


def _settings(path: str | None) -> Settings:
    return Settings.load(path)


def _train(args) -> int:
    from .train import train

    result = train(
        data=args.data,
        label=args.label,
        settings=_settings(args.config),
        features=args.features,
        time_col=args.time_col,
        id_col=args.id_col,
        run_name=args.name,
        registry_root=args.runs,
    )
    _print_training(result)
    if args.promote:
        from .registry import Registry

        Registry(args.runs).promote(result.run)
    return 0


def _score(args) -> int:
    from .scoring import score_batch

    result = score_batch(
        data=args.data,
        run=args.run,
        registry_root=args.runs,
        settings=_settings(args.config),
        output=args.output,
    )
    print(f"\nScored {len(result.scores)} row(s) with run '{result.run_name}'.")
    print(result.decision_mix().to_string())
    if result.alerts:
        print("\nDrift warnings:")
        for message in result.alerts:
            print(f"  - {message}")
    if not args.output:
        print("\nFirst rows:")
        print(result.scores.head(10).to_string(index=False))
    return 0


def _promote(args) -> int:
    from .registry import Registry

    run = Registry(args.runs).promote(args.run)
    print(f"Production is now '{run.name}'.")
    return 0


def _runs(args) -> int:
    from .registry import Registry

    registry = Registry(args.runs)
    runs = registry.list_runs()
    if not runs:
        print(f"No runs in {registry.root}.")
        return 0
    try:
        production = registry.production().name
    except (FileNotFoundError, ValueError):
        production = None

    rows = []
    for run in runs:
        summary = run.summary()
        metrics = {m["dataset"]: m for m in summary.get("model", {}).get("metrics", []) if m.get("model") == summary.get("model", {}).get("algorithm")}
        rows.append(
            {
                "run": run.name + (" *" if run.name == production else ""),
                "model": summary.get("model", {}).get("algorithm", "?"),
                "holdout_ks": round(metrics.get("holdout", {}).get("ks", float("nan")), 4),
                "rules": summary.get("rules", {}).get("stable", 0),
                "reject_rate": summary.get("outcome_on_holdout", {}).get("reject_rate"),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    if production:
        print("\n* production")
    return 0


def _compare(args) -> int:
    from .compare import compare_runs

    result = compare_runs(args.baseline, args.candidate, registry_root=args.runs, sample=args.sample)
    print(result.to_text())
    return 0


def _explain(args) -> int:
    from .registry import Registry

    run = Registry(args.runs).resolve(args.run)
    bundle = run.load_bundle()
    df = pd.read_csv(args.data)
    if not 0 <= args.row < len(df):
        raise ValueError(f"row {args.row} is out of range for a file with {len(df)} rows")
    row = df.iloc[[args.row]]
    scored = bundle.score(row).iloc[0]

    print(f"Run: {run.name}   Model: {bundle.metadata.get('algorithm', '?')}")
    print(f"Decision: {scored['decision'].upper()}  ({scored['reason']})")
    print(f"Model score {scored['model_score']:.4f} | expected bad rate {scored['calibrated_prob']:.2%} | credit score {scored['credit_score']:.0f}")
    print(f"Thresholds in force: decline at {scored['reject_at']:.4f}, review at {scored['review_at']:.4f}"
          + (f" (segment {scored['segment']})" if scored["segment"] else ""))

    print("\nHow each feature landed:")
    rows = []
    for feature in bundle.woe.features:
        binning = bundle.woe.binnings[feature]
        index = int(binning.assign(row[feature])[0])
        stat = next((s for s in binning.stats if s.index == index), None)
        rows.append(
            {
                "feature": feature,
                "value": row[feature].iloc[0],
                "bin": stat.label if stat else "?",
                "bin_bad_rate": stat.bad_rate if stat else None,
                "woe": stat.woe if stat else None,
            }
        )
    table = pd.DataFrame(rows)
    print(table.reindex(table["woe"].abs().sort_values(ascending=False).index).to_string(index=False))
    return 0


def _demo(args) -> int:
    from .data.synth import write_sample
    from .registry import Registry
    from .train import train

    path = Path(args.out)
    write_sample(str(path), n_rows=args.rows)
    print(f"Wrote {args.rows} synthetic applications to {path}.\n")

    settings = Settings().merged(
        {
            "split": {"time_col": "apply_time", "oot_months": 3},
            "cutoffs": {"segment_features": ["channel", "city_tier"]},
        }
    )
    result = train(
        data=path,
        label="is_bad",
        settings=settings,
        id_col="application_id",
        run_name=args.name,
        registry_root=args.runs,
    )
    _print_training(result)
    Registry(args.runs).promote(result.run)
    print(f"\nPromoted '{result.run.name}'. Try:")
    print(f"  riskflow score --data {path} --output scored.csv")
    print(f"  riskflow explain --data {path} --row 0")
    print(f"  open {result.run.report_path}")
    return 0


def _print_training(result) -> None:
    summary = result.summary
    model = summary["model"]
    outcome = summary["outcome_on_holdout"]
    print(f"\nRun '{summary['run']}' complete — {result.run.path}")
    print(f"  model            {model['algorithm']} ({model['backend']})")
    metrics = pd.DataFrame(model["metrics"])
    print("\n" + metrics[metrics["model"] == model["algorithm"]].to_string(index=False))
    print(f"\n  features kept    {len(summary['features']['selected'])} of {summary['features']['considered']}")
    print(f"  stable rules     {summary['rules']['stable']} of {summary['rules']['mined']} mined")
    print(f"  overfit verdict  {model['diagnostics'].get('overfit_verdict', 'n/a')}")
    if outcome.get("reject_rate") is not None:
        print(
            f"\n  On the sealed holdout: declining {outcome['reject_rate']:.1%} of applicants catches "
            f"{outcome['bad_capture_at_reject']:.1%} of the bads and leaves an approved bad rate of "
            f"{outcome['approved_bad_rate']:.2%}."
        )
    print(f"\n  report           {result.run.report_path}")


if __name__ == "__main__":
    sys.exit(main())
