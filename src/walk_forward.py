"""Walk-forward train/validation/test and parameter-selection utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Any, Iterable

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
    selected_parameters: list[dict[str, Any]] = field(default_factory=list)
    selection_results: list[pd.DataFrame] = field(default_factory=list)

    @property
    def validation_metrics(self) -> pd.DataFrame:
        return pd.DataFrame([r.metrics for r in self.validation_results])

    @property
    def test_metrics(self) -> pd.DataFrame:
        return pd.DataFrame([r.metrics for r in self.test_results])

    @property
    def selection_metrics(self) -> pd.DataFrame:
        frames = []
        for window, frame in enumerate(self.selection_results, start=1):
            current = frame.copy()
            current.insert(0, "window", window)
            frames.append(current)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class WalkForwardEvaluator:
    """Chronological expanding-window evaluator with explicit leakage boundaries.

    With ``parameter_grid``, every candidate is created from TRAIN only and
    scored on VALIDATION. The best candidate is then frozen and evaluated once
    on TEST. TEST is never used for parameter selection.
    """

    def __init__(
        self,
        train_size: int,
        validation_size: int,
        test_size: int,
        step_size: int | None = None,
    ) -> None:
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
        if not signals.index.equals(history.index):
            raise ValueError("time-series signal index must match history index")
        return signals.loc[evaluation_index]

    @staticmethod
    def _parameter_combinations(parameter_grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
        if not parameter_grid:
            raise ValueError("parameter_grid cannot be empty")
        keys = list(parameter_grid)
        values = [list(parameter_grid[key]) for key in keys]
        if any(not options for options in values):
            raise ValueError("parameter_grid values cannot be empty")
        return [dict(zip(keys, combo)) for combo in product(*values)]

    @staticmethod
    def _metric_value(result: BacktestResult, metric: str) -> float:
        if metric not in result.metrics:
            raise ValueError(f"selection metric not available: {metric}")
        value = float(result.metrics[metric])
        if pd.isna(value):
            return float("-inf")
        return value

    def _run_candidate(
        self,
        strategy: Any,
        history: pd.DataFrame,
        evaluation_index: pd.DatetimeIndex,
        evaluation_prices: pd.DataFrame,
        backtest_factory: Callable[[], BacktestEngine] | None,
    ) -> BacktestResult:
        signals = self._signals(strategy, history, evaluation_index)
        engine = backtest_factory() if backtest_factory else BacktestEngine()
        return engine.run(evaluation_prices, _SeriesSignalStrategy(signals, strategy))

    def evaluate(
        self,
        prices: pd.DataFrame,
        strategy_factory: Callable[[pd.DataFrame, dict[str, Any]], Any] | Callable[[pd.DataFrame], Any],
        backtest_factory: Callable[[], BacktestEngine] | None = None,
        parameter_grid: dict[str, Iterable[Any]] | None = None,
        selection_metric: str = "sharpe",
        maximize: bool = True,
    ) -> WalkForwardResult:
        """Evaluate windows, optionally selecting parameters on validation only.

        When ``parameter_grid`` is supplied, ``strategy_factory(train, params)``
        is called once per candidate. Each candidate sees TRAIN only. The
        selected instance is reused unchanged for TEST. The default objective
        is validation Sharpe; callers should choose the objective deliberately
        and avoid excessive parameter grids.
        """
        prices = prices.sort_index()
        windows = self.windows(prices.index)
        validation_results: list[BacktestResult] = []
        test_results: list[BacktestResult] = []
        selected_parameters: list[dict[str, Any]] = []
        selection_results: list[pd.DataFrame] = []
        rows = []

        candidates = self._parameter_combinations(parameter_grid) if parameter_grid is not None else [None]

        for number, window in enumerate(windows, start=1):
            train = prices.loc[window.train_start : window.train_end]
            validation = prices.loc[window.validation_start : window.validation_end]
            test = prices.loc[window.test_start : window.test_end]

            validation_history = pd.concat([train, validation])
            candidate_rows = []
            candidate_objects = []
            for params in candidates:
                try:
                    strategy = strategy_factory(train, params) if params is not None else strategy_factory(train)  # type: ignore[misc]
                    result = self._run_candidate(strategy, validation_history, validation.index, validation, backtest_factory)
                    score = self._metric_value(result, selection_metric)
                    candidate_rows.append({**(params or {}), selection_metric: score, "status": "ok"})
                    candidate_objects.append((params or {}, strategy, result, score))
                except Exception as exc:
                    candidate_rows.append({**(params or {}), selection_metric: float("-inf"), "status": f"error: {exc}"})

            if not candidate_objects:
                raise RuntimeError(f"no valid parameter candidate in window {number}")

            selected = max(candidate_objects, key=lambda item: item[3]) if maximize else min(candidate_objects, key=lambda item: item[3])
            selected_params, frozen_strategy, validation_result, selected_score = selected
            selection_frame = pd.DataFrame(candidate_rows).sort_values(selection_metric, ascending=not maximize, ignore_index=True)

            # TEST is evaluated only after validation selection and uses the
            # exact strategy object selected above. No refit or reselection.
            test_history = pd.concat([train, validation, test])
            test_result = self._run_candidate(frozen_strategy, test_history, test.index, test, backtest_factory)

            validation_results.append(validation_result)
            test_results.append(test_result)
            selected_parameters.append(selected_params)
            selection_results.append(selection_frame)
            rows.append({
                "window": number,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "validation_start": window.validation_start,
                "validation_end": window.validation_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_parameters": selected_params,
                "validation_selection_score": selected_score,
            })

        return WalkForwardResult(pd.DataFrame(rows), validation_results, test_results, selected_parameters, selection_results)


class _SeriesSignalStrategy:
    """Adapter exposing a precomputed point-in-time signal series to BacktestEngine."""

    def __init__(self, signals: pd.Series, source_strategy: Any) -> None:
        self.signals = signals
        self.config = getattr(source_strategy, "config", None)

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if not prices.index.equals(self.signals.index):
            raise ValueError("signal index must exactly match price index")
        return self.signals
