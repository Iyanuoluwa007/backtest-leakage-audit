"""Signal functions: three that leak the future and one that does not.

Each function takes a prices DataFrame and returns a Series of positions
(+1 long, -1 short, 0 flat) on the same index.
"""

import numpy as np
import pandas as pd


def momentum_signal(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Clean: long when price is above its trailing moving average."""
    ma = prices["close"].rolling(window).mean()
    return pd.Series(np.where(prices["close"] > ma, 1.0, -1.0), index=prices.index).where(ma.notna())


def centered_ma_signal(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Leaky: mean reversion to a centred moving average, which includes 10 future days."""
    ma = prices["close"].rolling(window, center=True).mean()
    return pd.Series(np.where(prices["close"] < ma, 1.0, -1.0), index=prices.index).where(ma.notna())


def full_sample_zscore_signal(prices: pd.DataFrame) -> pd.Series:
    """Leaky: the mean and standard deviation come from the entire history."""
    close = prices["close"]
    z = (close - close.mean()) / close.std()
    return pd.Series(np.where(z < 0, 1.0, -1.0), index=prices.index)


def confirmation_filter_signal(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Leaky: a 'trend confirmation' filter written with shift(-1) instead of shift(1)."""
    base = momentum_signal(prices, window)
    confirming_move = prices["close"].pct_change().shift(-1)  # meant to be shift(1)
    return base.where(np.sign(confirming_move) == base, 0.0)
