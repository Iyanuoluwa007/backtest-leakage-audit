"""Part 3: repeat the grid search on 100 independent noise histories.

One history can be lucky or unlucky. A hundred cannot.
Takes about a minute on a laptop.
"""

import numpy as np
import pandas as pd

from backtest_leakage_audit.data import make_prices, make_trending_prices
from backtest_leakage_audit.engine import sharpe, strategy_returns
from backtest_leakage_audit.overfitting import (
    deflated_sharpe_ratio,
    grid_search,
    out_of_sample_returns,
    parameter_grid,
    probabilistic_sharpe_ratio,
    sma_crossover_signal,
)

grid = parameter_grid()
rows = []
for seed in range(100):
    prices = make_prices(n_days=1008, seed=seed)
    in_sample = prices.iloc[:756]
    results = grid_search(in_sample, grid)
    best = results.iloc[0]
    fast, slow = int(best["fast"]), int(best["slow"])
    best_returns = strategy_returns(in_sample, sma_crossover_signal(in_sample, fast, slow))
    rows.append(
        {
            "in_sample": best["sharpe"],
            "out_of_sample": sharpe(out_of_sample_returns(prices, fast, slow, 756, 1008)),
            "psr": probabilistic_sharpe_ratio(best_returns),
            "dsr": deflated_sharpe_ratio(best_returns, results["daily_sharpe"].tolist()),
        }
    )

mc = pd.DataFrame(rows)
print("Grid-search winners on 100 noise histories (true Sharpe = 0)\n")
print(f"Mean in-sample Sharpe:      {mc['in_sample'].mean():.2f}")
print(f"Mean out-of-sample Sharpe:  {mc['out_of_sample'].mean():.2f}")
print(f"Winners with PSR > 0.95:    {(mc['psr'] > 0.95).mean():.0%}  (fooled by luck)")
print(f"Winners with DSR > 0.95:    {(mc['dsr'] > 0.95).mean():.0%}")

print("\nControl: one pre-registered idea on data with a real edge (N = 1, so DSR = PSR)\n")
for seed in [1, 2, 42]:
    prices = make_trending_prices(n_days=1008, seed=seed, phi=0.1)
    signal = np.sign(prices["close"].pct_change())  # yesterday up: go long
    rets = strategy_returns(prices, signal)
    in_sample, hidden = rets.iloc[:755], rets.iloc[755:]
    print(
        f"seed {seed:>2}: in-sample Sharpe {sharpe(in_sample):.2f}, "
        f"DSR {probabilistic_sharpe_ratio(in_sample):.3f}, hidden-year Sharpe {sharpe(hidden):.2f}"
    )
