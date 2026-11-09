# Changelog

## 2.0.0

### Changed (breaking)

- The code is now an installable package: `pip install backtest-leakage-audit`.
- Imports change from `from audit import audit_backtest` to `from backtest_leakage_audit import audit_backtest`.
- The tutorial scripts moved to `examples/`, and `audit.py`'s demo became `examples/audit_demo.py`.
- `requirements.txt` and `requirements-dev.txt` are replaced by `pyproject.toml`. Use `pip install -e ".[dev]"`.

### Added

- A `backtest-leakage-audit` command that audits a signal against a CSV of prices. Its exit code can gate CI.
- Automatic publishing to PyPI from a GitHub release, using trusted publishing.

### Fixed

- `lookahead_test` no longer crashes when reporting a leak on prices without a date index.

All golden outputs are unchanged, apart from `audit.txt` being renamed to `audit_demo.txt`.

## 1.1.0

### Fixed

- `lookahead_test` no longer crashes on histories under 301 rows. On a short history it now shrinks
  `min_history` and `n_checks` to fit, and it gives a clear error below 3 rows.
- `lookahead_test` no longer crashes when a signal returns fewer rows than the prices, as `.dropna()` does.
- The future-scramble test now rewrites every column, so a leak through `volume`, `high` or `low` is caught.

### Added

- `n_checks=None` tests every eligible row, for leaks that only touch a few rows.
- `audit_backtest(cost_bps=...)` audits the strategy after transaction costs.
- `snooping.py`: Hansen's SPA test, built on the `arch` package, resampling the full matrix of trial returns.
- `audit_backtest(trial_returns=...)` lets the SPA test decide the multiple-testing check.

### Changed

- Supported Python is now 3.11 to 3.15, because 3.10 reached end of life. On 3.15 the SPA test needs `arch`
  built from source, since it has no 3.15 wheel yet.
- CI runs weekly, and Dependabot watches the actions and dependencies.

All golden outputs in `expected_output/` are unchanged, so every number from 1.0.0 still reproduces.

## 1.0.0

First release.
