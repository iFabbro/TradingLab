import pandas as pd

from src.signals import TradeSetup, generate_setup


def _sample_df():
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "volume": [1000, 1100, 1050, 1200, 1300, 1250, 1400, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850],
        }
    )


def test_generate_setup_returns_dataclass():
    setup = generate_setup(_sample_df(), ticker="TEST", direction="long")
    assert isinstance(setup, TradeSetup)
    assert setup.entry > 0
    assert setup.stop < setup.entry
    assert setup.target > setup.entry
    assert setup.risk_reward > 0


def test_generate_setup_rejects_invalid_direction():
    try:
        generate_setup(_sample_df(), ticker="TEST", direction="flat")
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_generate_setup_uses_regime_decision():
    setup = generate_setup(_sample_df(), ticker="TEST", direction="long", regime_name="bullish_high_vol")
    assert setup.regime == "bullish_high_vol"
    assert setup.strategy_tag == "momentum_long"


def test_generate_setup_defaults_to_unknown_regime():
    setup = generate_setup(_sample_df(), ticker="TEST", direction="long")
    assert setup.regime == "unknown"
