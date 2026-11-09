"""A bootstrap data-snooping test: did the best of many backtested rules really beat zero?

This is Hansen's (2005) Test for Superior Predictive Ability, a sharper version of White's
(2000) Reality Check. The deflated Sharpe ratio only sees each trial's Sharpe ratio. This test
resamples the full matrix of trial returns, so correlation between trials is handled for you.

The `arch` package does the bootstrap. It is imported on first use, so the other checks still
run on a Python version that `arch` does not support yet.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import strategy_returns
from .overfitting import sma_crossover_signal


@dataclass
class SnoopingReport:
    n_trials: int
    p_value: float  # Hansen's consistent SPA p-value
    reality_check_p: float  # White's more conservative version


def trial_return_matrix(
    prices: pd.DataFrame, grid: list[tuple[int, int]], cost_bps: float = 0.0
) -> pd.DataFrame:
    """Daily returns of every SMA crossover in the grid, one column per parameter pair."""
    columns = {
        f"{fast}/{slow}": strategy_returns(prices, sma_crossover_signal(prices, fast, slow), cost_bps)
        for fast, slow in grid
    }
    return pd.DataFrame(columns).dropna()


def data_snooping_test(
    trial_returns: pd.DataFrame, block_size: int = 10, reps: int = 1000, seed: int = 0
) -> SnoopingReport:
    """Test whether the best column of trial_returns beats zero after allowing for the search.

    trial_returns needs one column of per-period returns for every strategy you tried, including
    the one you picked. A small p-value means at least the best of them has a real edge.
    block_size is the mean length of the resampled blocks, in rows.
    """
    try:
        from arch.bootstrap import SPA
    except ImportError as error:
        raise ImportError("data_snooping_test needs the arch package: pip install arch") from error

    returns = trial_returns.dropna()
    returns = returns.loc[:, returns.std(ddof=1) > 0]
    if returns.empty:
        raise ValueError("trial_returns has no rows or columns left after removing NaNs and flat strategies")

    # arch works with losses, so a strategy's loss is its negative return against a benchmark of zero
    test = SPA(
        np.zeros(len(returns)),
        -returns.to_numpy(),
        block_size=block_size,
        reps=reps,
        bootstrap="stationary",
        studentize=True,
        seed=seed,
    )
    test.compute()
    return SnoopingReport(returns.shape[1], float(test.pvalues["consistent"]), float(test.pvalues["upper"]))
