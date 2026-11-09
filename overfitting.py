"""Overfitting checks: the deflated Sharpe ratio and walk-forward validation.

Formulas follow Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".
Every Sharpe ratio inside these formulas is per-period (daily), not annualised.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from engine import TRADING_DAYS, sharpe, strategy_returns

EULER_GAMMA = 0.5772156649015329


# ---------- statistics ----------


def daily_sharpe(returns: pd.Series) -> float:
    return sharpe(returns, periods_per_year=1)


def annualise(daily_sr: float) -> float:
    return daily_sr * np.sqrt(TRADING_DAYS)


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probability that the true daily Sharpe ratio is above benchmark_sr."""
    sr = daily_sharpe(returns)
    t = len(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=False)  # raw kurtosis: 3.0 for a normal distribution
    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(t - 1) / denominator))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """The best daily Sharpe you should expect from n_trials strategies with zero skill."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(best_returns: pd.Series, trial_daily_sharpes: list[float]) -> float:
    """PSR of the winning strategy, measured against the luck hurdle instead of zero."""
    hurdle = expected_max_sharpe(len(trial_daily_sharpes), float(np.var(trial_daily_sharpes, ddof=1)))
    return probabilistic_sharpe_ratio(best_returns, benchmark_sr=hurdle)


# ---------- the strategy family we search over ----------


def parameter_grid() -> list[tuple[int, int]]:
    fasts = range(2, 52, 2)  # 25 fast windows
    slows = range(20, 220, 10)  # 20 slow windows
    return [(f, s) for f, s in itertools.product(fasts, slows) if f < s]


def sma_crossover_signal(prices: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    close = prices["close"]
    fast_ma, slow_ma = close.rolling(fast).mean(), close.rolling(slow).mean()
    return pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=prices.index).where(slow_ma.notna())


def grid_search(prices: pd.DataFrame, grid: list[tuple[int, int]], cost_bps: float = 0.0) -> pd.DataFrame:
    """Backtest every parameter pair and rank by annualised Sharpe."""
    rows = []
    for fast, slow in grid:
        rets = strategy_returns(prices, sma_crossover_signal(prices, fast, slow), cost_bps)
        rows.append({"fast": fast, "slow": slow, "sharpe": sharpe(rets), "daily_sharpe": daily_sharpe(rets)})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def out_of_sample_returns(
    prices: pd.DataFrame, fast: int, slow: int, test_start: int, test_end: int, cost_bps: float = 0.0
) -> pd.Series:
    """Returns on rows [test_start, test_end) only, with enough earlier rows to warm up the averages."""
    window = prices.iloc[max(0, test_start - slow) : test_end]
    rets = strategy_returns(window, sma_crossover_signal(window, fast, slow), cost_bps)
    return rets.loc[prices.index[test_start] :]


def walk_forward(
    prices: pd.DataFrame,
    grid: list[tuple[int, int]],
    train_days: int = 756,
    test_days: int = 252,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Optimise on each training window, then trade the winner on the next, unseen window."""
    folds = []
    start = 0
    while start + train_days + test_days <= len(prices):
        test_start, test_end = start + train_days, start + train_days + test_days
        best = grid_search(prices.iloc[start:test_start], grid, cost_bps).iloc[0]
        fast, slow = int(best["fast"]), int(best["slow"])
        oos = out_of_sample_returns(prices, fast, slow, test_start, test_end, cost_bps)
        folds.append(
            {
                "test_year_starts": prices.index[test_start].date(),
                "chosen": f"{fast}/{slow}",
                "in_sample": round(float(best["sharpe"]), 2),
                "out_of_sample": round(sharpe(oos), 2),
            }
        )
        start += test_days
    return pd.DataFrame(folds)
