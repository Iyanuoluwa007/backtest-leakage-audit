"""The future-scramble test: a black-box check for look-ahead bias.

Idea: rewrite everything after a cut point with fresh random prices.
Anything that was computed for a day before the cut must not change.
If it does, the computation read data from the future.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE_COLUMNS = {"open", "high", "low", "adj_close", "adj close", "adjclose", "vwap"}


@dataclass
class LookaheadReport:
    name: str
    checked: int
    mismatches: int
    examples: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.mismatches == 0


def scramble_future(prices: pd.DataFrame, cut: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy where every column is rewritten from row `cut` onward.

    `close` follows a brand-new random walk. Other price columns are rescaled to stay
    consistent with it, and everything else (volume, for example) has its future shuffled.
    """
    scrambled = prices.copy()
    n_future = len(prices) - cut
    start_price = prices["close"].iloc[cut - 1]
    new_path = start_price * np.exp(np.cumsum(rng.normal(0, 0.02, n_future)))
    scrambled.iloc[cut:, scrambled.columns.get_loc("close")] = new_path
    ratio = new_path / prices["close"].to_numpy()[cut:]
    for column in prices.columns.drop("close"):
        loc = scrambled.columns.get_loc(column)
        future = prices[column].to_numpy()[cut:]
        if str(column).lower() in PRICE_COLUMNS:
            scrambled[column] = scrambled[column].astype(float)
            scrambled.iloc[cut:, loc] = future * ratio
        else:
            scrambled.iloc[cut:, loc] = rng.permutation(future)
    return scrambled


def _on_index(values: pd.Series, index: pd.Index) -> pd.Series:
    """Put fn's output back on the price index, so dropped rows compare as NaN instead of shifting."""
    return values if values.index.equals(index) else values.reindex(index)


def same_value(a, b, atol: float = 1e-10) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(np.isclose(a, b, atol=atol))


def lookahead_test(
    fn: Callable[[pd.DataFrame], pd.Series],
    prices: pd.DataFrame,
    decision_lag: int = 0,
    n_checks: int | None = 50,
    min_history: int = 250,
    seed: int = 0,
) -> LookaheadReport:
    """Check that fn's output at a given row never depends on later rows.

    decision_lag=0: a value for day t may use data up to and including day t (signals, features).
    decision_lag=1: a value for day t may only use data up to day t - 1 (positions you hold on day t).

    n_checks=None tests every eligible row, which catches leaks that only touch a few rows.
    On a short history, min_history shrinks to half the rows and n_checks to the rows available.
    """
    rng = np.random.default_rng(seed)
    original = _on_index(fn(prices), prices.index)
    candidates = np.arange(min(min_history, len(prices) // 2), len(prices) - 1)
    if candidates.size == 0:
        raise ValueError(f"lookahead_test needs at least 3 rows of prices, got {len(prices)}")
    size = candidates.size if n_checks is None else min(n_checks, candidates.size)
    cut_points = rng.choice(candidates, size=size, replace=False)

    mismatches, examples = 0, []
    for cut in sorted(cut_points):
        # the last row that must be immune to changes from row `cut` onward
        row = cut - 1 + decision_lag
        altered = _on_index(fn(scramble_future(prices, cut, rng)), prices.index)
        if not same_value(original.iloc[row], altered.iloc[row]):
            mismatches += 1
            if len(examples) < 3:
                examples.append((prices.index[row].date(), original.iloc[row], altered.iloc[row]))

    return LookaheadReport(getattr(fn, "__name__", "fn"), len(cut_points), mismatches, examples)
