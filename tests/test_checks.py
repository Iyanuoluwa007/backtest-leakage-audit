"""Tests that prove the checks catch what they claim to catch."""

import numpy as np
import pytest

from backtest_leakage_audit.data import make_prices, make_trending_prices
from backtest_leakage_audit.engine import leaky_positions, positions, strategy_returns
from backtest_leakage_audit.lookahead import lookahead_test
from backtest_leakage_audit.overfitting import expected_max_sharpe, probabilistic_sharpe_ratio
from backtest_leakage_audit.signals import (
    centered_ma_signal,
    confirmation_filter_signal,
    full_sample_zscore_signal,
    momentum_signal,
)

PRICES = make_prices(n_days=756)


def test_clean_signal_passes():
    assert lookahead_test(momentum_signal, PRICES).passed


@pytest.mark.parametrize("fn", [centered_ma_signal, full_sample_zscore_signal, confirmation_filter_signal])
def test_leaky_signals_fail(fn):
    assert not lookahead_test(fn, PRICES).passed


def test_lagged_positions_pass():
    assert lookahead_test(lambda p: positions(momentum_signal(p)), PRICES, decision_lag=1).passed


def test_unlagged_positions_fail():
    assert not lookahead_test(lambda p: leaky_positions(momentum_signal(p)), PRICES, decision_lag=1).passed


def test_expected_max_sharpe_grows_with_trials():
    values = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(0)
    n_trials, n_obs, n_runs = 100, 252, 400
    maxima = []
    for _ in range(n_runs):
        r = rng.normal(0, 0.01, size=(n_trials, n_obs))
        srs = r.mean(axis=1) / r.std(axis=1, ddof=1)
        maxima.append(srs.max())
    theory = expected_max_sharpe(n_trials, 1 / n_obs)  # Var(SR) is about 1/T under the null
    assert abs(np.mean(maxima) - theory) / theory < 0.05


def test_psr_rewards_a_real_edge():
    p = make_trending_prices(n_days=756, seed=2)
    rets = strategy_returns(p, np.sign(p["close"].pct_change()))
    assert probabilistic_sharpe_ratio(rets) > 0.95
