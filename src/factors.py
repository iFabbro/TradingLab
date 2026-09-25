"""Price-based quantitative factors used by the research strategies.

These are intentionally named after the information they actually use. The
project does not currently claim to implement fundamental valuation factors.
"""
import numpy as np
import pandas as pd


def momentum_score(prices: pd.Series, window: int = 20) -> float:
    """Normalized trailing price momentum; positive means stronger momentum."""
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(prices) < window + 1:
        return 0.0
    ret = prices.iloc[-1] / prices.iloc[-window - 1] - 1
    return float(np.clip(ret / 0.2, -1, 1))


def mean_deviation_score(prices: pd.Series, window: int = 60) -> float:
    """Price-vs-moving-average mean-deviation score, not a fundamental value factor.

    Positive values indicate price below its trailing mean; negative values
    indicate price above it.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(prices) < window:
        return 0.0
    mean = prices.rolling(window).mean().iloc[-1]
    if not np.isfinite(mean) or mean == 0:
        return 0.0
    dev = (mean - prices.iloc[-1]) / mean
    return float(np.clip(dev / 0.2, -1, 1))


def volatility_score(prices: pd.Series, window: int = 20) -> float:
    """Low realized price volatility = higher score."""
    if window <= 0:
        raise ValueError("window must be > 0")
    if len(prices) < window + 1:
        return 0.0
    rets = prices.pct_change().dropna()
    vol = rets.iloc[-window:].std() * np.sqrt(252)
    score = 1 - vol / 0.8
    return float(np.clip(score, -1, 1))


def trend_score(prices: pd.Series, fast: int = 20, slow: int = 60) -> float:
    """Price trend score from fast-vs-slow moving averages."""
    if fast <= 0 or slow <= 0 or fast > slow:
        raise ValueError("fast and slow must be positive and fast <= slow")
    if len(prices) < slow:
        return 0.0
    sma_fast = prices.rolling(fast).mean().iloc[-1]
    sma_slow = prices.rolling(slow).mean().iloc[-1]
    if not np.isfinite(sma_slow) or sma_slow == 0:
        return 0.0
    dev = (sma_fast - sma_slow) / sma_slow
    return float(np.clip(dev / 0.1, -1, 1))
