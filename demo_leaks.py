"""Part 1: four signals on pure noise, and the test that exposes the leaks."""

from data import make_prices
from engine import leaky_positions, positions, sharpe, strategy_returns
from lookahead import lookahead_test
from signals import (
    centered_ma_signal,
    confirmation_filter_signal,
    full_sample_zscore_signal,
    momentum_signal,
)

SIGNALS = [momentum_signal, centered_ma_signal, full_sample_zscore_signal, confirmation_filter_signal]

prices = make_prices()

print("Annualised Sharpe on a driftless random walk (true Sharpe = 0)\n")
for fn in SIGNALS:
    print(f"{fn.__name__:<28}{sharpe(strategy_returns(prices, fn(prices))):>8.2f}")

sig = momentum_signal(prices)
print("\nThe clean momentum signal through two engines\n")
print(f"{'positions (lagged)':<28}{sharpe(strategy_returns(prices, sig)):>8.2f}")
print(
    f"{'leaky_positions (no lag)':<28}{sharpe(strategy_returns(prices, sig, position_fn=leaky_positions)):>8.2f}"
)

print("\nLook-ahead test on each signal (decision_lag=0)\n")
for fn in SIGNALS:
    r = lookahead_test(fn, prices, decision_lag=0)
    print(f"{'[OK]  ' if r.passed else '[FAIL]'} {r.name:<28} {r.mismatches}/{r.checked} values changed")

print("\nLook-ahead test on the positions from each engine (decision_lag=1)\n")
for name, position_fn in [("positions", positions), ("leaky_positions", leaky_positions)]:

    def held(p, position_fn=position_fn):
        return position_fn(momentum_signal(p))

    held.__name__ = name
    r = lookahead_test(held, prices, decision_lag=1)
    print(f"{'[OK]  ' if r.passed else '[FAIL]'} {r.name:<28} {r.mismatches}/{r.checked} values changed")
