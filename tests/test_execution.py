import pandas as pd
from scripts.run_bot import _load_risk_check
from src.execution import ExecutionEngine


def test_execution_engine_builds_order_intent():
    engine = ExecutionEngine()
    signal = {"ticker": "TEST", "side": "long"}
    risk_check = {"allowed": True}
    order = engine.build_order(signal=signal, risk_check=risk_check, position_size=2)

    assert isinstance(order, dict)
    assert order["ticker"] == "TEST"
    assert order["side"] == "long"
    assert order["quantity"] == 2
    assert order["status"] == "pending"


def test_execution_engine_blocks_when_risk_disallows():
    engine = ExecutionEngine()
    signal = {"ticker": "TEST", "side": "long"}
    risk_check = {"allowed": False}
    order = engine.build_order(signal=signal, risk_check=risk_check, position_size=2)

    assert order["ticker"] == "TEST"
    assert order["side"] == "long"
    assert order["quantity"] == 0
    assert order["status"] == "blocked"


def test_load_risk_check_accepts_csv(tmp_path):
    path = tmp_path / "risk_summary.csv"
    pd.DataFrame(
        [
            {
                "n_trades": 10,
                "win_rate": 0.5,
                "profit_factor": 1.5,
                "avg_rr": 1.2,
                "sharpe": 0.8,
                "sortino": 1.1,
                "warning_low_pf": False,
                "warning_nonpositive_sharpe": False,
            }
        ]
    ).to_csv(path, index=False)

    risk_check = _load_risk_check(str(path))

    assert risk_check["warning_low_pf"] is False
    assert risk_check["warning_nonpositive_sharpe"] is False
    assert risk_check["allowed"] is True
