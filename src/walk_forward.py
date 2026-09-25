"""Walk-forward train/validation/test evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import pandas as pd

from src.backtest import BacktestEngine, BacktestResult


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    validation_results: list[BacktestResult] = field(default_factory=list)
    test_results: list[BacktestResult] = field(default_factory=list)

    @property
    def validation_metrics(self) -> pd.DataFrame:
        return pd.DataFrame([r.metrics for r in self.validation_results])

    @property
    def test_metrics(self) -> pd.DataFrame:
        return pd.DataFrame([r.metrics for r in self.test_results])


class WalkForwardEvaluator:
    """Chronological expanding-window evaluator with explicit leakage boundaries.

    The strategy factory receives training data only. The returned strategy is
    frozen, then evaluated on validation and test slices without refitting.
    Historical observations preceding each evaluation slice are supplied only
    to the signal generator so indicators have the correct warm-up context.
    """

    def __init__(self, train_size: int, validation_size: int, test_size: int, step_size: int | None = None) -> None:
        if min(train_size, validation_size, test_size) <= 0:
            raise ValueError("window sizes must be > 0")
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")

    def windows(self, index: pd.DatetimeIndex) -> list[WalkForwardWindow]:
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex")
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("index must be unique and sorted")
        total = self.train_size + self.validation_size + self.test_size
        if len(index) < total:
            return []
        result = []
        start = 0
        while start + total <= len(index):
            train = index[start : start + self.train_size]
            validation = index[start + self.train_size : start + self.train_size + self.validation_size]
            test = index[start + self.train_size + self.validation_size : start + total]
            result.append(WalkForwardWindow(train[0], train[-1], validation[0], validation[-1], test[0], test[-1]))
            start += self.step_size
        return result

    @staticmethod
    def _signals(strategy: Any, history: pd.DataFrame, evaluation_index: pd.DatetimeIndex) -> pd.Series:
        generator = getattr(strategy, "generate_time_series_signals", None)
        if generator is None:
            raise TypeError("strategy must implement generate_time_series_signals for walk-forward evaluation")
        signals = generator(history)
        return signals.loc[evaluation_index]

    def evaluate(self, prices: pd.DataFrame, strategy_factory: Callable[[pd.DataFrame], Any], backtest_factory: Callable[[], BacktestEngine] | None = None) -> WalkForwardResult:
        prices = prices.sort_index()
        windows = self.windows(prices.index)
        validation_results: list[BacktestResult] = []
        test_results: list[BacktestResult] = []
        rows = []

        for number, window in enumerate(windows, start=1):
            train = prices.loc[window.train_start : window.train_end]
            validation = prices.loc[window.validation_start : window.validation_end]
            test = prices.loc[window.test_start : window.test_end]

            strategy = strategy_factory(train)
            validation_history = pd.concat([train, validation])
            validation_signals = self._signals(strategy, validation_history, validation.index)
            validation_engine = backtest_factory() if backtest_factory else BacktestEngine()
            validation_result = validation_engine.run(validation, _SeriesSignalStrategy(validation_signals, strategy))

            test_history = pd.concat([train, validation, test])
            test_signals = self._signals(strategy, test_history, test.index)
            test_engine = backtest_factory() if backtest_factory else BacktestEngine()
            test_result = test_engine.run(test, _SeriesSignalStrategy(test_signals, strategy))

            validation_results.append(validation_result)
            test_results.append(test_result)
            rows.append({
                "window": number,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "validation_start": window.validation_start,
                "validation_end": window.validation_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
            })

        return WalkForwardResult(pd.DataFrame(rows), validation_results, test_results)


class _SeriesSignalStrategy:
    """Adapter exposing a precomputed point-in-time signal series to BacktestEngine."""

    def __init__(self, signals: pd.Series, source_strategy: Any) -> None:
        self.signals = signals
        self.config = getattr(source_strategy, "config", None)

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if not prices.index.equals(self.signals.index):
            raise ValueError("signal index must exactly match price index")
        return self.signals
