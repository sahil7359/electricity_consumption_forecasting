"""Command-line interface.

Makes the analysis reproducible without opening a notebook or a browser - useful
in CI, and the fastest way to check that a change to a model actually moved the
numbers.
"""

from __future__ import annotations

import argparse
import sys

from elecforecast.analysis import adf_test, make_stationary
from elecforecast.data import describe, load_series
from elecforecast.evaluate import compare, walk_forward
from elecforecast.models import MODELS, build


def cmd_info(args: argparse.Namespace) -> int:
    series = load_series()
    facts = describe(series)
    print("Electric production series")
    for key, value in facts.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<14} {value}")

    report = adf_test(series)
    print(f"\nADF on raw series: {report.verdict}")
    _, n_diffs, after = make_stationary(series)
    print(f"Differences needed: d = {n_diffs}")
    print(f"After differencing: {after.verdict}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    series = load_series()
    model = build(args.model).fit(series)
    forecast = model.predict(args.horizon)

    print(f"{model.name} - {args.horizon} month forecast\n")
    for timestamp, value in forecast.items():
        print(f"  {timestamp:%Y-%m}  {value:8.2f}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    series = load_series()

    if args.model:
        result = walk_forward(
            MODELS[args.model],
            series,
            horizon=args.horizon,
            n_folds=args.folds,
            label=args.model,
        )
        print(f"{build(args.model).name} - per fold\n")
        print(result.folds.to_string(index=False))
        print(f"\nMean: {result.summary}")
        print(
            "\nBeats the seasonal-naive baseline."
            if result.beats_baseline
            else "\nDoes not beat the seasonal-naive baseline."
        )
        return 0

    table = compare(MODELS, series, horizon=args.horizon, n_folds=args.folds)
    print(f"Walk-forward backtest - {args.folds} folds x {args.horizon} months\n")
    print(table.to_string(index=False))
    print("\nMASE below 1.0 beats the seasonal-naive baseline; above 1.0 loses to it.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elecforecast",
        description="Forecast monthly electric utility production.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="dataset summary and stationarity diagnostics")
    info.set_defaults(func=cmd_info)

    forecast = sub.add_parser("forecast", help="forecast forward from the full series")
    forecast.add_argument("--model", choices=sorted(MODELS), default="sarima")
    forecast.add_argument("--horizon", type=int, default=12, help="months ahead")
    forecast.set_defaults(func=cmd_forecast)

    backtest = sub.add_parser("backtest", help="walk-forward evaluation")
    backtest.add_argument(
        "--model", choices=sorted(MODELS), default=None, help="omit to compare all"
    )
    backtest.add_argument("--horizon", type=int, default=12)
    backtest.add_argument("--folds", type=int, default=5)
    backtest.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
