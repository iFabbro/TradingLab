from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable

from src.backtest import BacktestEngine, BacktestResult
from src.strategies import BaseStrategy, StrategyConfig


@dataclass(frozen=True)
class OptimizationCandidate:
    params: dict[str, Any]
    train_result: BacktestResult
    test_result: BacktestResult
    score: float


@dataclass(frozen=True)
class OptimizationResult:
    baseline_train: BacktestResult
    baseline_test: BacktestResult
    candidates: list[OptimizationCandidate]

    @property
    def best_candidate(self) -> OptimizationCandidate | None:
        return max(self.candidates, key=lambda c: c.score) if self.candidates else None


def _build_param_grid(param_grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    if not keys:
        return [{}]
    values = [list(param_grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _merge_config(base_config: StrategyConfig, overrides: dict[str, Any]) -> StrategyConfig:
    data = base_config.__dict__.copy()
    data.update(overrides)
    return StrategyConfig(**data)


def optimize_strategy(
    engine_factory: Callable[[StrategyConfig], BacktestEngine],
    strategy_factory: Callable[[StrategyConfig], BaseStrategy],
    base_config: StrategyConfig,
    train_data: Any,
    test_data: Any,
    param_grid: dict[str, Iterable[Any]],
    metric_name: str = "sharpe",
    min_test_over_baseline: float = 0.0,
    min_train_test_ratio: float = 0.7,
    max_drawdown_gap: float = 0.10,
) -> OptimizationResult:
    baseline_engine = engine_factory(base_config)
    baseline_strategy = strategy_factory(base_config)
    baseline_train = baseline_engine.run(train_data, baseline_strategy)
    baseline_test = baseline_engine.run(test_data, baseline_strategy)

    baseline_metric = float(baseline_train.metrics.get(metric_name, 0.0))
    candidates: list[OptimizationCandidate] = []

    for params in _build_param_grid(param_grid):
        config = _merge_config(base_config, params)
        engine = engine_factory(config)
        strategy = strategy_factory(config)
        train_result = engine.run(train_data, strategy)
        test_result = engine.run(test_data, strategy)

        train_metric = float(train_result.metrics.get(metric_name, 0.0))
        test_metric = float(test_result.metrics.get(metric_name, 0.0))
        train_dd = float(train_result.metrics.get("max_drawdown", 0.0))
        test_dd = float(test_result.metrics.get("max_drawdown", 0.0))

        if train_metric <= baseline_metric:
            continue
        if baseline_metric > 0 and test_metric < baseline_metric + min_test_over_baseline:
            continue
        if train_metric == 0:
            continue
        if test_metric / train_metric < min_train_test_ratio:
            continue
        if test_dd - train_dd > max_drawdown_gap:
            continue

        score = test_metric + 0.5 * train_metric
        candidates.append(
            OptimizationCandidate(
                params=params,
                train_result=train_result,
                test_result=test_result,
                score=score,
            )
        )

    return OptimizationResult(
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        candidates=candidates,
    )


def format_optimization_report(result: OptimizationResult) -> str:
    lines = []
    lines.append("Baseline train:")
    lines.append(f"  {result.baseline_train.metrics}")
    lines.append("Baseline test:")
    lines.append(f"  {result.baseline_test.metrics}")

    if not result.candidates:
        lines.append("No candidate passed anti-overfitting filters.")
        return "\n".join(lines)

    best = result.best_candidate
    lines.append("Best candidate:")
    lines.append(f"  params={best.params}")
    lines.append(f"  train={best.train_result.metrics}")
    lines.append(f"  test={best.test_result.metrics}")
    lines.append(f"  score={best.score:.6f}")
    return "\n".join(lines)
