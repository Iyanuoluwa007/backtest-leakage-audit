"""Command-line audit of a signal function against a CSV of prices.

Exit code 0 means every check passed, 1 means at least one failed, and 2 means bad input,
so the command can gate a CI job.
"""

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .audit import audit_backtest


def _load_signal(spec: str) -> Callable[[pd.DataFrame], pd.Series]:
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"--signal must look like module:function, got {spec!r}")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module_name), attribute)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtest-leakage-audit",
        description="Audit a trading signal for look-ahead bias and overfitting.",
    )
    parser.add_argument("prices", help="CSV with dates in the first column and a 'close' column")
    parser.add_argument("--signal", required=True, metavar="MODULE:FUNCTION", help="signal function to audit")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="transaction cost per unit traded")
    parser.add_argument("--all-rows", action="store_true", help="test every eligible row, not a sample of 50")
    parser.add_argument(
        "--trials",
        metavar="CSV",
        help="daily returns of every strategy you tried, one column each, including this one",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        prices = _read_csv(args.prices)
        signal_fn = _load_signal(args.signal)
        trial_returns = _read_csv(args.trials) if args.trials else None
    except (OSError, ImportError, AttributeError, ValueError) as error:
        parser.error(str(error))
    if "close" not in prices.columns:
        parser.error(f"{args.prices} has no 'close' column")

    try:
        passed = audit_backtest(
            prices,
            signal_fn,
            cost_bps=args.cost_bps,
            n_checks=None if args.all_rows else 50,
            trial_returns=trial_returns,
        )
    except ImportError as error:  # the SPA test needs arch, which may not be installed
        parser.error(str(error))
    return 0 if passed else 1
