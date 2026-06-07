"""
Fase 5 — Market Regime Detection
Classifica il contesto di mercato su tre dimensioni: trend, volatilità, volume.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Costanti di default (override via parametri)
# ---------------------------------------------------------------------------
DEFAULT_FAST = 20
DEFAULT_SLOW = 50
DEFAULT_VOL_WINDOW = 20
DEFAULT_VOL_THRESHOLD_HIGH = 1.5
DEFAULT_VOL_THRESHOLD_LOW = 0.5
DEFAULT_VOLUME_WINDOW = 20
DEFAULT_VOLUME_THRESHOLD_HIGH = 1.5
DEFAULT_VOLUME_THRESHOLD_LOW = 0.5


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
def classify_trend(
    close: pd.Series,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
) -> pd.Series:
    """Restituisce una Series con valori: 'bullish', 'bearish', 'sideways'."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    trend = pd.Series("sideways", index=close.index, dtype=object)
    trend[sma_fast > sma_slow] = "bullish"
    trend[sma_fast < sma_slow] = "bearish"
    trend[sma_fast.isna() | sma_slow.isna()] = "unknown"
    return trend


# ---------------------------------------------------------------------------
# Volatilità
# ---------------------------------------------------------------------------
def classify_volatility(
    close: pd.Series,
    window: int = DEFAULT_VOL_WINDOW,
    threshold_high: float = DEFAULT_VOL_THRESHOLD_HIGH,
    threshold_low: float = DEFAULT_VOL_THRESHOLD_LOW,
) -> pd.Series:
    """Restituisce una Series con valori: 'high_vol', 'low_vol', 'normal_vol'."""
    returns = close.pct_change()
    rolling_std = returns.rolling(window).std()
    mean_std = rolling_std.expanding().mean()
    vol = pd.Series("normal_vol", index=close.index, dtype=object)
    vol[rolling_std > threshold_high * mean_std] = "high_vol"
    vol[rolling_std < threshold_low * mean_std] = "low_vol"
    vol[rolling_std.isna()] = "unknown"
    return vol


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
def classify_volume(
    volume: pd.Series,
    window: int = DEFAULT_VOLUME_WINDOW,
    threshold_high: float = DEFAULT_VOLUME_THRESHOLD_HIGH,
    threshold_low: float = DEFAULT_VOLUME_THRESHOLD_LOW,
) -> pd.Series:
    """Restituisce una Series con valori: 'high_volume', 'low_volume', 'normal_volume'."""
    rolling_vol = volume.rolling(window).mean()
    vol_cls = pd.Series("normal_volume", index=volume.index, dtype=object)
    vol_cls[volume > threshold_high * rolling_vol] = "high_volume"
    vol_cls[volume < threshold_low * rolling_vol] = "low_volume"
    vol_cls[rolling_vol.isna()] = "unknown"
    return vol_cls


# ---------------------------------------------------------------------------
# Regime complessivo
# ---------------------------------------------------------------------------
def detect_regime(
    close: pd.Series,
    volume: pd.Series | None = None,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    vol_window: int = DEFAULT_VOL_WINDOW,
    volume_window: int = DEFAULT_VOLUME_WINDOW,
) -> pd.DataFrame:
    """
    Combina trend, volatilità e volume in un DataFrame con colonne:
        trend, volatility, volume_regime, regime
    regime = "{trend}_{volatility}" (es. "bullish_high_vol")
    """
    trend = classify_trend(close, fast=fast, slow=slow)
    volatility = classify_volatility(close, window=vol_window)

    df = pd.DataFrame({"trend": trend, "volatility": volatility}, index=close.index)

    if volume is not None:
        df["volume_regime"] = classify_volume(volume, window=volume_window)
    else:
        df["volume_regime"] = "no_data"

    df["regime"] = df["trend"] + "_" + df["volatility"]
    return df


# ---------------------------------------------------------------------------
# Mapping regime → strategia consigliata
# ---------------------------------------------------------------------------
REGIME_STRATEGY_MAP: dict[str, str] = {
    "bullish_high_vol":   "momentum_long",
    "bullish_normal_vol": "trend_following_long",
    "bullish_low_vol":    "breakout_long",
    "bearish_high_vol":   "momentum_short",
    "bearish_normal_vol": "trend_following_short",
    "bearish_low_vol":    "breakout_short",
    "sideways_high_vol":  "mean_reversion",
    "sideways_normal_vol":"range_trading",
    "sideways_low_vol":   "neutral_wait",
    "unknown_unknown":    "no_trade",
}


def suggest_strategy(regime: str) -> str:
    """Restituisce la strategia suggerita per un dato regime."""
    return REGIME_STRATEGY_MAP.get(regime, "undefined")
