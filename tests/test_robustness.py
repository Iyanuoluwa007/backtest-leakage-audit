"""Tests for the v1.1 fixes: short histories, every column scrambled, rare leaks, costs and SPA."""

import numpy as np
import pandas as pd
import pytest

from audit import audit_backtest
from data import make_prices, make_trending_prices
from engine import strategy_returns
from lookahead import lookahead_test, scramble_future
from overfitting import (
    daily_sharpe,
    deflated_sharpe_ratio,
    parameter_grid,
    probabilistic_sharpe_ratio,
    sma_crossover_signal,
)
from signals import centered_ma_signal, momentum_signal
from snooping import data_snooping_test, trial_return_matrix

SMALL_GRID = parameter_grid()[::5]


def with_ohlcv(prices: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = prices.copy()
    frame["high"] = frame["close"] * (1 + np.abs(rng.normal(0, 0.005, len(frame))))
    frame["low"] = frame["close"] * (1 - np.abs(rng.normal(0, 0.005, len(frame))))
    frame["volume"] = rng.lognormal(10, 0.5, len(frame))
    return frame


def one_day_momentum(prices: pd.DataFrame) -> pd.Series:
    return np.sign(prices["close"].pct_change())


# ---------- short or uneven output ----------


@pytest.mark.parametrize("n_days", [60, 252, 290, 301])
def test_short_histories_run(n_days):
    assert lookahead_test(momentum_signal, make_prices(n_days=n_days)).passed


def test_short_history_still_catches_a_leak():
    assert not lookahead_test(centered_ma_signal, make_prices(n_days=252)).passed


def test_too_few_rows_gives_a_clear_error():
    with pytest.raises(ValueError, match="at least 3 rows"):
        lookahead_test(momentum_signal, make_prices(n_days=2))


def test_clean_signal_with_dropped_rows_passes():
    assert lookahead_test(lambda p: momentum_signal(p).dropna(), make_prices(n_days=756)).passed


def test_leaky_signal_with_dropped_rows_fails():
    assert not lookahead_test(lambda p: centered_ma_signal(p).dropna(), make_prices(n_days=756)).passed


# ---------- every column is scrambled ----------


def test_scramble_leaves_the_past_alone_and_rewrites_the_future():
    frame = with_ohlcv(make_prices(n_days=400))
    scrambled = scramble_future(frame, 300, np.random.default_rng(0))
    pd.testing.assert_frame_equal(scrambled.iloc[:300], frame.iloc[:300])
    for column in frame.columns:
        assert not np.allclose(scrambled[column].iloc[300:], frame[column].iloc[300:]), column


def test_scramble_keeps_high_and_low_around_close():
    frame = with_ohlcv(make_prices(n_days=400))
    future = scramble_future(frame, 300, np.random.default_rng(0)).iloc[300:]
    assert (future["high"] >= future["close"]).all() and (future["low"] <= future["close"]).all()


def test_scramble_handles_integer_price_columns():
    frame = make_prices(n_days=400)
    frame["open"] = frame["close"].round().astype(int)
    scrambled = scramble_future(frame, 300, np.random.default_rng(0))
    assert not np.allclose(scrambled["open"].iloc[300:], frame["open"].iloc[300:])


@pytest.mark.parametrize(
    "leak",
    [
        lambda p: np.sign(p["volume"].shift(-1) - p["volume"]),
        lambda p: np.sign(p["high"].shift(-1) - p["close"]),
        lambda p: np.sign(p["close"] - p["low"].shift(-1)),
    ],
    ids=["volume", "high", "low"],
)
def test_leaks_in_other_columns_fail(leak):
    assert not lookahead_test(leak, with_ohlcv(make_prices(n_days=756))).passed


def test_honest_use_of_other_columns_passes():
    def volume_trend(p):
        return np.sign(p["volume"] - p["volume"].rolling(20).mean())

    assert lookahead_test(volume_trend, with_ohlcv(make_prices(n_days=756))).passed


# ---------- exhaustive mode ----------


def rare_leak(prices: pd.DataFrame) -> pd.Series:
    """A clean signal that peeks at tomorrow on one row in every 100."""
    signal = momentum_signal(prices).copy()
    peek = np.sign(prices["close"].pct_change().shift(-1))
    mask = np.arange(len(prices)) % 100 == 0
    signal[mask] = peek[mask]
    return signal


def test_exhaustive_mode_catches_a_rare_leak():
    assert not lookahead_test(rare_leak, make_prices(n_days=756), n_checks=None).passed


def test_exhaustive_mode_checks_every_eligible_row():
    report = lookahead_test(momentum_signal, make_prices(n_days=756), n_checks=None)
    assert report.passed and report.checked == 756 - 1 - 250


# ---------- statistics ----------


def test_dsr_is_stricter_than_psr():
    prices = make_trending_prices(n_days=756, seed=2)
    rets = strategy_returns(prices, one_day_momentum(prices))
    trials = np.random.default_rng(0).normal(0, 0.03, 100).tolist()
    assert deflated_sharpe_ratio(rets, trials) < probabilistic_sharpe_ratio(rets)


def test_dsr_matches_psr_when_trials_do_not_differ():
    prices = make_trending_prices(n_days=756, seed=2)
    rets = strategy_returns(prices, one_day_momentum(prices))
    assert deflated_sharpe_ratio(rets, [0.1, 0.1]) == pytest.approx(probabilistic_sharpe_ratio(rets))


# ---------- costs ----------


def test_costs_can_turn_a_pass_into_a_fail(capsys):
    prices = make_trending_prices(n_days=756, seed=2)
    assert audit_backtest(prices, one_day_momentum)
    assert not audit_backtest(prices, one_day_momentum, cost_bps=50)
    assert "after 50 bps costs" in capsys.readouterr().out


# ---------- bootstrap data-snooping test ----------


def edge_trials(n_days: int = 2520, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = make_trending_prices(n_days=n_days, seed=seed)
    trials = trial_return_matrix(prices, SMALL_GRID)
    trials["real"] = strategy_returns(prices, one_day_momentum(prices))
    return prices, trials.dropna()


def test_spa_finds_a_real_edge_inside_a_search():
    pytest.importorskip("arch")
    assert data_snooping_test(edge_trials()[1]).p_value < 0.05


def test_spa_does_not_find_an_edge_in_noise():
    pytest.importorskip("arch")
    trials = trial_return_matrix(make_prices(n_days=756), SMALL_GRID)
    report = data_snooping_test(trials)
    assert report.p_value >= 0.05 and report.n_trials == len(SMALL_GRID)


def test_spa_reports_a_conservative_reality_check():
    pytest.importorskip("arch")
    report = data_snooping_test(trial_return_matrix(make_prices(n_days=756), SMALL_GRID))
    assert report.reality_check_p >= report.p_value


def test_spa_ignores_flat_strategies():
    pytest.importorskip("arch")
    trials = trial_return_matrix(make_prices(n_days=756), SMALL_GRID[:5])
    trials["flat"] = 0.0
    assert data_snooping_test(trials).n_trials == 5


def test_spa_rejects_empty_input():
    pytest.importorskip("arch")
    with pytest.raises(ValueError, match="no rows or columns"):
        data_snooping_test(pd.DataFrame({"flat": [0.0, 0.0, 0.0]}))


def test_audit_uses_spa_and_passes_a_real_edge(capsys):
    pytest.importorskip("arch")
    prices, trials = edge_trials()
    assert audit_backtest(prices, one_day_momentum, trial_returns=trials)
    assert "SPA p =" in capsys.readouterr().out


def test_audit_uses_spa_and_fails_noise():
    pytest.importorskip("arch")
    prices = make_prices(n_days=756)
    trials = trial_return_matrix(prices, SMALL_GRID)
    best = trials.apply(daily_sharpe).idxmax()
    fast, slow = (int(part) for part in best.split("/"))

    def winner(p):
        return sma_crossover_signal(p, fast, slow)

    assert not audit_backtest(prices, winner, trial_returns=trials)
