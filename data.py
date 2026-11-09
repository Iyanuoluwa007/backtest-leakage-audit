"""Synthetic price data.

make_prices: a driftless random walk. Every strategy tested on it has a true
Sharpe ratio of zero, so any profit you see is a bug or luck.

make_trending_prices: returns with a small positive autocorrelation, so trend
following has a real (modest) edge. Use it as a control: a good check should
pass this data, not just fail everything.
"""

import numpy as np
import pandas as pd


def _to_frame(log_returns: np.ndarray) -> pd.DataFrame:
    close = 100 * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range("2015-01-01", periods=len(log_returns), name="date")
    return pd.DataFrame({"close": close}, index=index)


def make_prices(n_days: int = 2520, seed: int = 42, daily_vol: float = 0.01) -> pd.DataFrame:
    """About 10 years of trading days with no edge to find."""
    rng = np.random.default_rng(seed)
    return _to_frame(rng.normal(0.0, daily_vol, n_days))


def make_trending_prices(
    n_days: int = 2520, seed: int = 42, daily_vol: float = 0.01, phi: float = 0.1
) -> pd.DataFrame:
    """Returns follow an AR(1) process: today's return echoes a fraction phi of yesterday's."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, daily_vol, n_days)
    returns = np.empty(n_days)
    returns[0] = shocks[0]
    for t in range(1, n_days):
        returns[t] = phi * returns[t - 1] + shocks[t]
    return _to_frame(returns)
