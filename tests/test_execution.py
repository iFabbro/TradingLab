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
