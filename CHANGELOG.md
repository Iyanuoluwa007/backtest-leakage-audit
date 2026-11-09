# Changelog

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
