"""Part 2: a clean engine and clean signals can still produce a fake edge."""

from backtest_leakage_audit.data import make_prices
from backtest_leakage_audit.engine import sharpe, strategy_returns
from backtest_leakage_audit.overfitting import (
    annualise,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    grid_search,
    out_of_sample_returns,
    parameter_grid,
    probabilistic_sharpe_ratio,
    sma_crossover_signal,
    walk_forward,
)

grid = parameter_grid()

# Four years of noise: optimise on the first three, keep the fourth hidden
prices = make_prices(n_days=1008, seed=42)
in_sample = prices.iloc[:756]

results = grid_search(in_sample, grid)
best = results.iloc[0]
fast, slow = int(best["fast"]), int(best["slow"])
best_returns = strategy_returns(in_sample, sma_crossover_signal(in_sample, fast, slow))

print(f"Tested {len(results)} SMA crossover pairs on 3 years of a driftless random walk\n")
print(results.head(5).to_string(index=False, float_format="%.3f"))

hurdle = expected_max_sharpe(len(results), results["daily_sharpe"].var(ddof=1))
print(f"\nWinner {fast}/{slow}: annualised Sharpe {best['sharpe']:.2f}")
print(f"Luck hurdle for {len(results)} trials:  {annualise(hurdle):.2f} (annualised)")
print(f"Probabilistic Sharpe ratio:     {probabilistic_sharpe_ratio(best_returns):.3f}")
dsr = deflated_sharpe_ratio(best_returns, results["daily_sharpe"].tolist())
print(f"Deflated Sharpe ratio:          {dsr:.3f}")

hidden_year = out_of_sample_returns(prices, fast, slow, 756, 1008)
print(f"Sharpe in the hidden fourth year: {sharpe(hidden_year):.2f}")

print("\nWalk-forward on 10 years: optimise on 3 years, trade the next year unseen\n")
wf = walk_forward(make_prices(seed=42), grid)
print(wf.to_string(index=False))
print(f"\nMean in-sample Sharpe:     {wf['in_sample'].mean():.2f}")
print(f"Mean out-of-sample Sharpe: {wf['out_of_sample'].mean():.2f}")
