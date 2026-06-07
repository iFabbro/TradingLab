import numpy as np
import pandas as pd
import pytest
from src.factors import momentum_score, value_score, volatility_score, trend_score
from src.strategies import MultiFactorStrategy


@pytest.fixture
def trending_up():
    return pd.Series(np.linspace(100, 150, 120))  # trend rialzista netto


@pytest.fixture
def flat():
    rng = np.random.default_rng(42)
    return pd.Series(100 + rng.normal(0, 0.5, 120))  # mercato piatto


def test_momentum_up(trending_up):
    assert momentum_score(trending_up) > 0


def test_momentum_range(trending_up):
    s = momentum_score(trending_up)
    assert -1.0 <= s <= 1.0


def test_value_score_range(trending_up):
    s = value_score(trending_up)
    assert -1.0 <= s <= 1.0


def test_volatility_low_on_flat(flat):
    # mercato piatto → volatilità bassa → score positivo
    assert volatility_score(flat) > 0


def test_trend_up(trending_up):
    assert trend_score(trending_up) > 0


def test_multifactor_score_range(trending_up):
    strat = MultiFactorStrategy()
    s = strat.score(trending_up)
    assert -1.0 <= s <= 1.0


def test_multifactor_signal_buy(trending_up):
    strat = MultiFactorStrategy()
    assert strat.signal(trending_up) == "BUY"


def test_multifactor_weights_normalize():
    strat = MultiFactorStrategy(weights={"momentum": 2, "value": 2, "volatility": 1, "trend": 1})
    assert abs(sum(strat.weights.values()) - 1.0) < 1e-9


def test_short_series_returns_zero():
    short = pd.Series([100, 101, 102])
    assert momentum_score(short) == 0.0
    assert trend_score(short) == 0.0
