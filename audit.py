"""Part 4: one function that runs every check and prints a plain-text report."""

from collections.abc import Callable

import pandas as pd

from engine import positions, sharpe, strategy_returns
from lookahead import lookahead_test
from overfitting import daily_sharpe, deflated_sharpe_ratio, probabilistic_sharpe_ratio
from snooping import data_snooping_test


def audit_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    position_fn: Callable[[pd.Series], pd.Series] = positions,
    trial_daily_sharpes: list[float] | None = None,
    dsr_threshold: float = 0.95,
    cost_bps: float = 0.0,
    n_checks: int | None = 50,
    trial_returns: pd.DataFrame | None = None,
    snooping_alpha: float = 0.05,
) -> bool:
    """Return True only if the backtest passes every check.

    cost_bps: transaction cost per unit traded, applied to the audited strategy.
    n_checks: look-ahead cut points to test. None tests every eligible row.
    trial_returns: daily returns of every strategy tried, one per column, including this one and
    after the same costs. When given, the bootstrap SPA test decides the multiple-testing check
    and the DSR is shown for reference.
    """
    checks = []

    sig_report = lookahead_test(signal_fn, prices, decision_lag=0, n_checks=n_checks)
    checks.append(
        (
            "signal uses no future data",
            sig_report.passed,
            f"{sig_report.mismatches}/{sig_report.checked} values changed",
        )
    )

    def held(p):
        return position_fn(signal_fn(p))

    pos_report = lookahead_test(held, prices, decision_lag=1, n_checks=n_checks)
    checks.append(
        (
            "position is decided before the bar it trades",
            pos_report.passed,
            f"{pos_report.mismatches}/{pos_report.checked} values changed",
        )
    )

    rets = strategy_returns(prices, signal_fn(prices), cost_bps=cost_bps, position_fn=position_fn)
    trials = trial_daily_sharpes or []
    if not trials and trial_returns is not None:
        trials = [daily_sharpe(trial_returns[column]) for column in trial_returns]
    if len(trials) > 1:
        score, label = deflated_sharpe_ratio(rets, trials), f"DSR over {len(trials)} trials"
    else:
        score, label = probabilistic_sharpe_ratio(rets), "PSR (1 trial)"
    edge_passed, edge_detail = score >= dsr_threshold, f"{label} = {score:.3f}"
    if trial_returns is not None:
        spa = data_snooping_test(trial_returns)
        edge_passed = spa.p_value < snooping_alpha
        edge_detail = f"SPA p = {spa.p_value:.3f}, {edge_detail}"
    checks.append(("edge survives multiple testing", edge_passed, edge_detail))

    cost_note = f" after {cost_bps:g} bps costs" if cost_bps else ""
    name = getattr(signal_fn, "__name__", "signal")
    print(f"Audit: {name}  (annualised Sharpe {sharpe(rets):.2f}{cost_note})")
    for name, passed, detail in checks:
        print(f"  {'[OK]  ' if passed else '[FAIL]'} {name:<46} {detail}")
    verdict = all(passed for _, passed, _ in checks)
    print(f"  Verdict: {'PASS' if verdict else 'DO NOT TRUST THIS BACKTEST'}\n")
    return verdict


if __name__ == "__main__":
    import numpy as np

    from data import make_prices, make_trending_prices
    from engine import leaky_positions
    from overfitting import grid_search, parameter_grid, sma_crossover_signal
    from signals import confirmation_filter_signal, momentum_signal

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
