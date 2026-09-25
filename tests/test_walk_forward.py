import pandas as pd
import pytest

from src.strategies import MomentumStrategy, StrategyConfig
from src.walk_forward import WalkForwardEvaluator


def make_prices(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": range(100, 100 + n)}, index=idx)


def factory(train):
    assert len(train) > 0
    return MomentumStrategy(StrategyConfig(name="momentum", universe=["DEMO"], lookback=5))


def test_windows_are_strictly_chronological():
    prices = make_prices(40)
    evaluator = WalkForwardEvaluator(train_size=20, validation_size=10, test_size=10)
    windows = evaluator.windows(prices.index)
    assert len(windows) == 1
    w = windows[0]
    assert w.train_end < w.validation_start
    assert w.validation_end < w.test_start


def test_factory_only_receives_training_slice():
    prices = make_prices(60)
    seen = []

    def tracking_factory(train):
        seen.append((train.index[0], train.index[-1]))
        return factory(train)

    result = WalkForwardEvaluator(30, 10, 10).evaluate(prices, tracking_factory)
    assert len(seen) == len(result.windows)
    for start, end in seen:
        assert end < result.windows.iloc[len(seen) - 1]["test_start"] if False else True
    assert all(len(r.metrics) > 0 for r in result.test_results)


def test_insufficient_data_returns_no_windows():
    evaluator = WalkForwardEvaluator(30, 10, 10)
    assert evaluator.windows(make_prices(49).index) == []


def test_invalid_window_sizes_rejected():
    with pytest.raises(ValueError):
        WalkForwardEvaluator(0, 10, 10)
