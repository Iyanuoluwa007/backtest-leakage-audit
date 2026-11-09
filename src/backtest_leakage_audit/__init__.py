"""Catch look-ahead bias and overfitting in trading backtests."""

from importlib.metadata import version

from .audit import audit_backtest
from .engine import leaky_positions, positions, sharpe, strategy_returns
from .lookahead import LookaheadReport, lookahead_test, scramble_future
from .overfitting import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    grid_search,
    parameter_grid,
    probabilistic_sharpe_ratio,
    walk_forward,
)
from .snooping import SnoopingReport, data_snooping_test, trial_return_matrix

__version__ = version("backtest-leakage-audit")

__all__ = [
    "LookaheadReport",
    "SnoopingReport",
    "__version__",
    "audit_backtest",
    "data_snooping_test",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "grid_search",
    "leaky_positions",
    "lookahead_test",
    "parameter_grid",
    "positions",
    "probabilistic_sharpe_ratio",
    "scramble_future",
    "sharpe",
    "strategy_returns",
    "trial_return_matrix",
    "walk_forward",
]
