"""Tests for the backtest-leakage-audit command."""

import sys

import pytest

from backtest_leakage_audit import __version__
from backtest_leakage_audit.cli import main
from backtest_leakage_audit.data import make_prices, make_trending_prices
from backtest_leakage_audit.engine import strategy_returns
from backtest_leakage_audit.overfitting import parameter_grid
from backtest_leakage_audit.snooping import trial_return_matrix

SIGNALS = """
import numpy as np

def honest(p):
    return np.sign(p["close"].pct_change())

def leaky(p):
    return np.sign(p["close"].pct_change().shift(-1))
"""


@pytest.fixture
def workspace(tmp_path, monkeypatch, request):
    """A folder with a prices CSV and a uniquely named signal module, used as the working directory."""
    make_trending_prices(n_days=756, seed=2).to_csv(tmp_path / "prices.csv")
    module = f"signals_{request.node.name}"
    (tmp_path / f"{module}.py").write_text(SIGNALS)
    monkeypatch.chdir(tmp_path)
    return module


def test_passing_audit_exits_zero(workspace):
    assert main(["prices.csv", "--signal", f"{workspace}:honest"]) == 0


def test_leaky_signal_exits_one(workspace, capsys):
    assert main(["prices.csv", "--signal", f"{workspace}:leaky"]) == 1
    assert "DO NOT TRUST" in capsys.readouterr().out


def test_cost_and_all_rows_options(workspace, capsys):
    assert main(["prices.csv", "--signal", f"{workspace}:honest", "--cost-bps", "50", "--all-rows"]) == 1
    output = capsys.readouterr().out
    assert "after 50 bps costs" in output and "/505 values changed" in output


def test_trials_file_switches_to_spa(workspace, tmp_path, capsys):
    pytest.importorskip("arch")
    prices = make_prices(n_days=756)
    prices.to_csv(tmp_path / "noise.csv")
    trial_return_matrix(prices, parameter_grid()[::25]).to_csv(tmp_path / "trials.csv")
    main(["noise.csv", "--signal", f"{workspace}:honest", "--trials", "trials.csv"])
    assert "SPA p =" in capsys.readouterr().out


def test_integer_index_works(workspace, tmp_path):
    frame = make_trending_prices(n_days=756, seed=2).reset_index(drop=True)
    frame.to_csv(tmp_path / "numbered.csv")
    assert main(["numbered.csv", "--signal", f"{workspace}:leaky"]) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["missing.csv", "--signal", "{module}:honest"],
        ["prices.csv", "--signal", "no_colon"],
        ["prices.csv", "--signal", "no_such_module:honest"],
        ["prices.csv", "--signal", "{module}:no_such_function"],
        ["no_close.csv", "--signal", "{module}:honest"],
    ],
    ids=["missing-file", "bad-spec", "bad-module", "bad-function", "no-close"],
)
def test_bad_input_exits_two(workspace, tmp_path, arguments):
    (tmp_path / "no_close.csv").write_text("date,open\n2020-01-01,1.0\n")
    with pytest.raises(SystemExit) as error:
        main([part.format(module=workspace) for part in arguments])
    assert error.value.code == 2


def test_missing_arch_exits_two(workspace, tmp_path, monkeypatch):
    prices = make_prices(n_days=756)
    prices.to_csv(tmp_path / "noise.csv")
    trial_return_matrix(prices, parameter_grid()[::25]).to_csv(tmp_path / "trials.csv")
    monkeypatch.setitem(sys.modules, "arch.bootstrap", None)
    with pytest.raises(SystemExit) as error:
        main(["noise.csv", "--signal", f"{workspace}:honest", "--trials", "trials.csv"])
    assert error.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0 and __version__ in capsys.readouterr().out


def test_strategy_returns_is_exported():
    import backtest_leakage_audit

    assert backtest_leakage_audit.strategy_returns is strategy_returns
