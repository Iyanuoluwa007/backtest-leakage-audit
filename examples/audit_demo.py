"""Part 4: run every check on four strategies and print one verdict each."""

import numpy as np

from backtest_leakage_audit.audit import audit_backtest
from backtest_leakage_audit.data import make_prices, make_trending_prices
from backtest_leakage_audit.engine import leaky_positions
from backtest_leakage_audit.overfitting import grid_search, parameter_grid, sma_crossover_signal
from backtest_leakage_audit.signals import confirmation_filter_signal, momentum_signal

noise = make_prices(n_days=756)

audit_backtest(noise, confirmation_filter_signal)
audit_backtest(noise, momentum_signal, position_fn=leaky_positions)

results = grid_search(noise, parameter_grid())
fast, slow = int(results.iloc[0]["fast"]), int(results.iloc[0]["slow"])


def best_crossover(p):
    return sma_crossover_signal(p, fast, slow)


audit_backtest(noise, best_crossover, trial_daily_sharpes=results["daily_sharpe"].tolist())


def one_day_momentum(p):
    return np.sign(p["close"].pct_change())


audit_backtest(make_trending_prices(n_days=756, seed=2), one_day_momentum)
