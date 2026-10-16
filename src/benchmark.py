"""Buy-and-hold benchmark computed on exactly the same out-of-sample days as a strategy.

An absolute out-of-sample return means little on its own. The research registry therefore
records, next to every strategy result, what a passive long position in the same instrument
would have earned over the *same* OOS windows.

Conventions (kept identical to ``WalkForwardResult.oos_daily_returns`` so the comparison is
like-for-like):

* each OOS window is evaluated on its own slice of closes;
* daily return = close-to-close change inside the window, and the first day of every window
  is 0.0 (``pct_change().fillna(0.0)``), exactly as for the strategy's equity curve;
* windows are pooled chronologically; overlapping windows are rejected;
* the passive position is 100% long and is entered once, on the first OOS day, paying the
  protocol's transaction cost and slippage once (a strategy pays them on every exposure change).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.statistics import sharpe_ratio

BENCHMARK_ID = "buy_and_hold_long_net_entry_cost"


def buy_and_hold_oos(
    prices: pd.DataFrame,
    windows: pd.DataFrame,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """Return buy-and-hold OOS total return and annualised Sharpe over the given test windows.

    ``prices`` needs a sorted ``DatetimeIndex`` and a ``close`` column. ``windows`` is the
    ``WalkForwardResult.windows`` frame (columns ``test_start`` and ``test_end``).
    """
    if "close" not in prices.columns:
        raise ValueError("prices must contain a 'close' column")
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_monotonic_increasing:
        raise ValueError("prices must have a sorted DatetimeIndex")
    if windows.empty:
        raise ValueError("no OOS windows to benchmark")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("costs must be non-negative")

    pieces = []
    for _, window in windows.iterrows():
        closes = prices["close"].loc[window["test_start"]:window["test_end"]]
        if closes.empty:
            raise ValueError(f"empty OOS slice for window starting {window['test_start']}")
        pieces.append(closes.pct_change().fillna(0.0))

    pooled = pd.concat(pieces)
    if pooled.index.has_duplicates:
        raise ValueError("OOS test windows overlap; cannot pool daily returns")
    pooled = pooled.sort_index().astype(float)

    entry_cost = (float(transaction_cost_bps) + float(slippage_bps)) / 10_000.0
    returns = pooled.copy()
    returns.iloc[0] -= entry_cost

    values = returns.to_numpy()
    return {
        "benchmark": BENCHMARK_ID,
        "observations": int(len(returns)),
        "start": returns.index.min().isoformat(),
        "end": returns.index.max().isoformat(),
        "entry_cost_bps": float(transaction_cost_bps) + float(slippage_bps),
        "total_return": float(np.prod(1.0 + values) - 1.0),
        "sharpe": float(sharpe_ratio(values, annualisation=252.0)),
    }
