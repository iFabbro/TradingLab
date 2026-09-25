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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name!r})"


class MomentumStrategy(BaseStrategy):
    """Select assets by cross-sectional trailing price momentum."""

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
