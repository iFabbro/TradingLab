from dataclasses import dataclass

from src.optimization import optimize_strategy
from src.strategies import StrategyConfig


@dataclass
class DummyResult:
    sharpe_ratio: float
    max_drawdown: float


class DummyEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def run(self, data):
        p = getattr(self.config, "lookback", 3)
        if data == "train":
            return DummyResult(sharpe_ratio=float(p), max_drawdown=0.10 + p * 0.01)
        return DummyResult(sharpe_ratio=float(p) - 0.5, max_drawdown=0.12 + p * 0.01)


def make_config():
    return StrategyConfig(name="dummy", universe=["SPY"], lookback=3)


def test_optimizer_keeps_only_valid_candidates():
    base = make_config()
    result = optimize_strategy(
        engine_factory=lambda cfg: DummyEngine(cfg),
        base_config=base,
        train_data="train",
        test_data="test",
        param_grid={"lookback": [2, 3, 4, 5]},
        min_test_over_baseline=-1.0,
        min_train_test_ratio=0.8,
        max_drawdown_gap=0.2,
    )

    assert result.baseline_train.sharpe_ratio == 3.0
    assert result.baseline_test.sharpe_ratio == 2.5
    assert len(result.candidates) >= 1
    assert all(candidate.params["lookback"] >= 4 for candidate in result.candidates)


def test_optimizer_rejects_overfitted_candidates():
    base = make_config()
    result = optimize_strategy(
        engine_factory=lambda cfg: DummyEngine(cfg),
        base_config=base,
        train_data="train",
        test_data="test",
        param_grid={"lookback": [6]},
        min_test_over_baseline=-1.0,
        min_train_test_ratio=0.9,
        max_drawdown_gap=0.01,
    )

    assert result.candidates == []
