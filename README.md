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

A control dataset with a real edge passes every check, so the tools are not simply rejecting everything.

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

## Project layout

| File | Purpose |
| --- | --- |
| `data.py` | Synthetic prices: a driftless random walk and an AR(1) control with a real edge |
| `engine.py` | A tiny vectorised backtest engine, plus the deliberately leaky version |
| `signals.py` | One honest signal and three that leak the future |
| `lookahead.py` | The future-scramble test for look-ahead bias |
| `overfitting.py` | Probabilistic and deflated Sharpe ratios, grid search, walk-forward |
| `audit.py` | Runs every check and prints a verdict |
| `demo_*.py`, `monte_carlo.py` | The scripts behind every number in the tutorial |
| `expected_output/` | Exact output of each script, checked by `tests/test_golden_outputs.py` |

## Reproducibility

Every number in the tutorial is checked in CI. The golden-output tests run each script and compare its
output byte for byte with `expected_output/`. Tested on Python 3.10 with pandas 2.x and Python 3.13 with
pandas 3.x. The one-minute Monte Carlo check runs with `RUN_SLOW=1 python -m pytest -q`.

## What these checks can't catch

Survivorship bias, data that was revised after publication, leaks in columns the scramble test does not
rewrite, an under-counted number of trials, and realistic execution costs. See the tutorial for details.

## Reference

Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*, 40(5), 94-107.
[SSRN 2460551](https://ssrn.com/abstract=2460551)

## License

[Apache-2.0](LICENSE)
