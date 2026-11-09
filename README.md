# backtest-leakage-audit

[![CI](https://github.com/Iyanuoluwa007/backtest-leakage-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Iyanuoluwa007/backtest-leakage-audit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Catch the two most common lies a trading backtest tells: **look-ahead bias** and **overfitting**.

This is the companion code for the tutorial
*How to Catch Look-Ahead Bias and Overfitting in Your Trading Backtests with Python* (link added once published).

Everything runs on synthetic data, so no market data, API key or broker account is needed.
On a driftless random walk the true Sharpe ratio of every strategy is zero, so any profit is a bug or luck.

## What it shows

| Problem | What goes wrong on pure noise | What catches it |
| --- | --- | --- |
| Signal reads the future (`shift(-1)`, centred windows, full-sample stats) | Sharpe ratios up to 10.81 | Future-scramble test on the signal |
| Engine trades on the same bar it decides (missing `shift(1)`) | Sharpe jumps from 0.52 to 4.86 | Future-scramble test on the positions |
| Picking the best of 466 parameter sets | Mean in-sample Sharpe 0.98, out-of-sample 0.08 | Deflated Sharpe ratio (fooled 0% of the time vs 42% for a plain test) |

A control dataset with a real edge passes every check when it is the only idea tested, so the tools are not
simply rejecting everything. Searching many ideas raises the bar: see [How much data you need](#how-much-data-you-need).

## Quick start

```bash
git clone https://github.com/Iyanuoluwa007/backtest-leakage-audit.git
cd backtest-leakage-audit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python demo_leaks.py         # leaky signals and the future-scramble test
python demo_overfitting.py   # grid search, deflated Sharpe ratio, walk-forward
python monte_carlo.py        # 100 noise histories plus a control (about 1 minute)
python audit.py              # one plain-text verdict per strategy
python -m pytest -q          # unit tests and golden-output tests
```

Sample audit report:

```plaintext
Audit: best_crossover  (annualised Sharpe 0.84)
  [OK]   signal uses no future data                     0/50 values changed
  [OK]   position is decided before the bar it trades   0/50 values changed
  [FAIL] edge survives multiple testing                 DSR over 466 trials = 0.215
  Verdict: DO NOT TRUST THIS BACKTEST
```

## Auditing your own strategy

`audit_backtest` takes a prices DataFrame with a `close` column, plus any other columns your signal reads.

```python
from audit import audit_backtest
from overfitting import parameter_grid
from snooping import trial_return_matrix

trials = trial_return_matrix(prices, parameter_grid(), cost_bps=5)  # or build your own: one column per idea

audit_backtest(
    prices,
    my_signal,
    cost_bps=5,  # audit after transaction costs
    n_checks=None,  # test every row, which catches leaks that only touch a few rows
    trial_returns=trials,  # every strategy you tried, including this one
)
```

Columns named `open`, `high`, `low`, `adj_close` or `vwap` are rescaled with the scrambled close. Every other
column has its future shuffled.

## How much data you need

Both multiple-testing checks need a long history once you have searched many ideas. Here is how often each one
passed in 15 to 30 synthetic histories, searching 466 moving-average pairs. The "Real edge" columns add one
rule that truly works to the search:

| Test (pass at 5%) | Noise, 3 years | Real edge, 3 years | Noise, 10 years | Real edge, 10 years |
| --- | --- | --- | --- | --- |
| Deflated Sharpe ratio | 3% | 7% | 0% | 73% |
| SPA test | 7% | 23% | 0% | 87% |

A failed check on a short history can mean "not enough evidence yet". It does not prove there is no edge.

## Project layout

| File | Purpose |
| --- | --- |
| `data.py` | Synthetic prices: a driftless random walk and an AR(1) control with a real edge |
| `engine.py` | A tiny vectorised backtest engine, plus the deliberately leaky version |
| `signals.py` | One honest signal and three that leak the future |
| `lookahead.py` | The future-scramble test for look-ahead bias |
| `overfitting.py` | Probabilistic and deflated Sharpe ratios, grid search, walk-forward |
| `snooping.py` | Hansen's SPA test across every strategy you tried |
| `audit.py` | Runs every check and prints a verdict |
| `demo_*.py`, `monte_carlo.py` | The scripts behind every number in the tutorial |
| `expected_output/` | Exact output of each script, checked by `tests/test_golden_outputs.py` |

## Reproducibility

Every number in the tutorial is checked in CI. The golden-output tests run each script and compare its
output byte for byte with `expected_output/`. Tested on Python 3.11 with pandas 2.x, and on Python 3.14 and
3.15 with pandas 3.x. The SPA tests skip on 3.15, where `arch` has no wheel yet. The one-minute Monte Carlo
check runs with `RUN_SLOW=1 python -m pytest -q`.

The tutorial's code matches the [`v1.0.0`](https://github.com/Iyanuoluwa007/backtest-leakage-audit/tree/v1.0.0)
tag. See the [changelog](CHANGELOG.md) for what has changed since.

## What these checks can't catch

Survivorship bias, data that was revised after publication, leaks from data outside the prices DataFrame, an
under-counted number of trials, and execution costs beyond a flat `cost_bps`. See the tutorial for details.

## Reference

Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*, 40(5), 94-107.
[SSRN 2460551](https://ssrn.com/abstract=2460551)

Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business and Economic Statistics*,
23(4), 365-380.

White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097-1126.

## License

[Apache-2.0](LICENSE)
