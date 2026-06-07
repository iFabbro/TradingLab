"""
Fase 9 — Multi-factor strategy
Ogni funzione riceve un pd.Series di prezzi e restituisce un float in [-1, 1].
"""
import pandas as pd
import numpy as np


def momentum_score(prices: pd.Series, window: int = 20) -> float:
    """Return normalizzato della finestra. Positivo = momentum rialzista."""
    if len(prices) < window + 1:
        return 0.0
    ret = prices.iloc[-1] / prices.iloc[-window - 1] - 1
    return float(np.clip(ret / 0.2, -1, 1))  # normalizza su ±20%


def value_score(prices: pd.Series, window: int = 60) -> float:
    """Distanza dal prezzo medio. Negativo = costoso, positivo = economico."""
    if len(prices) < window:
        return 0.0
    mean = prices.rolling(window).mean().iloc[-1]
    dev = (mean - prices.iloc[-1]) / mean
    return float(np.clip(dev / 0.2, -1, 1))


def volatility_score(prices: pd.Series, window: int = 20) -> float:
    """Volatilità bassa = score alto. Normalizzata su annualizzata 0-80%."""
    if len(prices) < window + 1:
        return 0.0
    rets = prices.pct_change().dropna()
    vol = rets.iloc[-window:].std() * np.sqrt(252)
    score = 1 - vol / 0.8  # vol 0% → 1, vol 80%+ → ≤0
    return float(np.clip(score, -1, 1))


def trend_score(prices: pd.Series, fast: int = 20, slow: int = 60) -> float:
    """SMA fast vs slow. Positivo = trend rialzista."""
    if len(prices) < slow:
        return 0.0
    sma_fast = prices.rolling(fast).mean().iloc[-1]
    sma_slow = prices.rolling(slow).mean().iloc[-1]
    dev = (sma_fast - sma_slow) / sma_slow
    return float(np.clip(dev / 0.1, -1, 1))  # normalizza su ±10%
