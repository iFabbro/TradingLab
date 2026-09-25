from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.factors import momentum_score, mean_deviation_score, volatility_score, trend_score


@dataclass
class StrategyConfig:
    name: str
    universe: list[str]
    lookback: int
    rebalance_freq: str = "monthly"
    long_only: bool = True
    top_n: Optional[int] = None
    params: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.universe:
            raise ValueError("universe cannot be empty")
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if self.rebalance_freq not in {"daily", "weekly", "monthly"}:
            raise ValueError(f"invalid rebalance_freq: {self.rebalance_freq}")
        if self.top_n is not None and not 0 < self.top_n <= len(self.universe):
            raise ValueError("top_n must be between 1 and universe size")


class BaseStrategy:
    """Common interface for strategies consumed by the backtest engine."""

    def __init__(self, config: StrategyConfig) -> None:
        config.validate()
        self.config = config

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def generate_time_series_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Generate point-in-time signals without using future observations.

        This adapter is for single-series research/backtesting. Multi-asset
        portfolio construction remains a separate responsibility.
        """
        if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices must be a non-empty DatetimeIndex DataFrame")
        signals = []
        for timestamp in prices.index:
            snapshot = prices.loc[:timestamp]
            result = self.generate_signals(snapshot)
            if "close" not in result.index:
                raise ValueError("time-series adapter requires a 'close' signal")
            signals.append(float(result.loc["close"]))
        return pd.Series(signals, index=prices.index, name="signal")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name!r})"


class MomentumStrategy(BaseStrategy):
    """Return trailing momentum scores, optionally restricted to the top-N assets.

    Contract: the returned Series is indexed by the input assets and contains
    continuous momentum scores. When ``top_n`` is set, only the selected
    assets retain their score and all other assets are exactly zero. The
    strategy does not normalize the surviving scores to binary selections or
    to portfolio weights; portfolio sizing is a separate layer.
    """

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        skip = max(0, int(self.config.params.get("skip_last", 1)))
        if len(prices) < self.config.lookback + skip:
            return pd.Series(0.0, index=prices.columns)
        end_idx = len(prices) - skip if skip else len(prices)
        start_idx = end_idx - self.config.lookback
        window = prices.iloc[start_idx:end_idx]
        scores = ((window.iloc[-1] / window.iloc[0]) - 1).replace([float("inf"), -float("inf")], 0).fillna(0.0)
        if self.config.top_n is not None:
            selected = scores.nlargest(self.config.top_n).index
            return scores.where(scores.index.isin(selected), 0.0)
        return scores


class DonchianBreakoutStrategy(BaseStrategy):
    """Long-only Donchian channel breakout with point-in-time state.

    For each asset, a long position is entered when today's close exceeds the
    previous ``lookback`` closes' maximum and is exited when today's close
    falls below the previous ``lookback`` closes' minimum. Between breakouts
    and exits the prior position is held. The current bar is never included in
    the channel used to generate its signal, preventing look-ahead bias.
    """

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            return pd.Series(0.0, index=prices.columns)
        lookback = int(self.config.lookback)
        if lookback <= 1:
            raise ValueError("Donchian lookback must be > 1")
        if len(prices) <= lookback:
            return pd.Series(0.0, index=prices.columns, dtype=float)

        signals = pd.Series(0.0, index=prices.columns, dtype=float)
        for column in prices.columns:
            series = pd.to_numeric(prices[column], errors="coerce")
            if series.isna().any() or not (series > 0).all():
                raise ValueError("Donchian prices must contain positive finite values")
            state = 0.0
            for i in range(lookback, len(series)):
                history = series.iloc[i - lookback:i]
                close = float(series.iloc[i])
                upper = float(history.max())
                lower = float(history.min())
                if close > upper:
                    state = 1.0
                elif close < lower:
                    state = 0.0
            signals.loc[column] = state
        return signals


class MeanReversionStrategy(BaseStrategy):
    """Generate long-only reversion scores from trailing price z-scores."""

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if len(prices) < self.config.lookback:
            return pd.Series(0.0, index=prices.columns)
        window = prices.iloc[-self.config.lookback:]
        mean = window.mean()
        std = window.std().replace(0, float("nan"))
        z = (prices.iloc[-1] - mean) / std
        threshold = float(self.config.params.get("z_threshold", 1.0))
        signal = -z
        signal[z.abs() < threshold] = 0.0
        if self.config.long_only:
            signal[signal < 0] = 0.0
        return signal.fillna(0.0)


class TrendFollowingStrategy(BaseStrategy):
    """Generate long signals when the fast SMA is above the slow SMA."""

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        fast = int(self.config.params.get("fast", 20))
        slow = int(self.config.params.get("slow", self.config.lookback))
        if fast <= 0 or slow <= 0 or fast > slow:
            raise ValueError("fast and slow must be positive and fast <= slow")
        if len(prices) < slow:
            return pd.Series(0.0, index=prices.columns)
        return (prices.iloc[-fast:].mean() > prices.iloc[-slow:].mean()).astype(float)


class MultiFactorStrategy:
    """Combine price momentum, mean deviation, volatility and trend scores."""

    DEFAULT_WEIGHTS = {"momentum": 0.4, "mean_deviation": 0.2, "volatility": 0.2, "trend": 0.2}

    def __init__(self, weights: dict | None = None):
        w = dict(weights or self.DEFAULT_WEIGHTS)
        if set(w) != set(self.DEFAULT_WEIGHTS):
            raise ValueError(f"weights must contain {set(self.DEFAULT_WEIGHTS)}")
        if any(value < 0 for value in w.values()):
            raise ValueError("weights must be >= 0")
        total = sum(w.values())
        if total <= 0:
            raise ValueError("weight sum must be > 0")
        self.weights = {k: v / total for k, v in w.items()}

    def score(self, prices: pd.Series) -> float:
        factors = {
            "momentum": momentum_score(prices),
            "mean_deviation": mean_deviation_score(prices),
            "volatility": volatility_score(prices),
            "trend": trend_score(prices),
        }
        return round(float(sum(self.weights[k] * v for k, v in factors.items())), 4)

    def signal(self, prices: pd.Series, threshold: float = 0.1) -> str:
        score = self.score(prices)
        if score > threshold:
            return "BUY"
        if score < -threshold:
            return "SELL"
        return "HOLD"
