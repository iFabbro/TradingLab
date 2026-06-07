from dataclasses import dataclass

from src.optimization import optimize_strategy
from src.strategies import StrategyConfig, BaseStrategy


@dataclass
class DummyResult:
    trade_log: object
    equity_curve: object
    metrics: dict


class DummyStrategy(BaseStrategy):
    def generate_signals(self, prices):
        return prices.iloc[-1] * 0


class DummyEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def run(self, data, strategy):
        p = getattr(self.config, "lookback", 3)
        metrics = {
            "sharpe": float(p) if data == "train" else float(p) - 0.5,
            "max_drawdown": 0.10 + p * 0.01 if data == "train" else 0.12 + p * 0.01,
        }
        return DummyResult(trade_log=None, equity_curve=None, metrics=metrics)


def make_config():
    return StrategyConfig(name="dummy", universe=["SPY"], lookback=3)


def make_strategy(cfg):
    return DummyStrategy(cfg)


def test_optimizer_keeps_only_valid_candidates():
    base = make_config()
    result = optimize_strategy(
        engine_factory=lambda cfg: DummyEngine(cfg),
        strategy_factory=make_strategy,
        base_config=base,
        train_data="train",
        test_data="test",
        param_grid={"lookback": [2, 3, 4, 5]},
        min_test_over_baseline=-1.0,
        min_train_test_ratio=0.8,
        max_drawdown_gap=0.2,
    )

    assert result.baseline_train.metrics["sharpe"] == 3.0
    assert result.baseline_test.metrics["sharpe"] == 2.5
    assert len(result.candidates) >= 1
    assert all(candidate.params["lookback"] >= 4 for candidate in result.candidates)


def test_optimizer_rejects_overfitted_candidates():
    base = make_config()
    result = optimize_strategy(
        engine_factory=lambda cfg: DummyEngine(cfg),
        strategy_factory=make_strategy,
        base_config=base,
        train_data="train",
        test_data="test",
        param_grid={"lookback": [6]},
        min_test_over_baseline=-1.0,
        min_train_test_ratio=0.9,
        max_drawdown_gap=0.01,
    )

    assert result.candidates == []
