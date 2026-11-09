"""A deliberately tiny vectorised backtest engine.

Convention: a signal computed from data up to the close of day t
becomes the position you hold during day t + 1.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def positions(signal: pd.Series) -> pd.Series:
    """Correct: what you hold today was decided at yesterday's close."""
    return signal.shift(1)


def leaky_positions(signal: pd.Series) -> pd.Series:
    """The most common bug: today's position uses today's close."""
    return signal


def strategy_returns(
    prices: pd.DataFrame, signal: pd.Series, cost_bps: float = 0.0, position_fn=positions
) -> pd.Series:
    """Daily strategy returns after optional transaction costs (in basis points per unit traded)."""
    asset_returns = prices["close"].pct_change()
    pos = position_fn(signal)
    costs = pos.diff().abs() * cost_bps / 10_000
    return (pos * asset_returns - costs).dropna()


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio (risk-free rate assumed to be zero)."""
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))
