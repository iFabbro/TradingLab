import json
import sys

import pandas as pd
import pytest

from scripts.run_bot import _load_risk_check
from src.execution import ExecutionEngine
from src.safety import SafetyPolicy, evaluate_safety


def test_execution_engine_builds_order_intent():
    order = ExecutionEngine().build_order({"ticker": "TEST", "side": "long"}, {"allowed": True}, 2)
    assert order == {"ticker": "TEST", "side": "long", "quantity": 2, "status": "pending"}


def test_execution_engine_blocks_when_risk_disallows():
    order = ExecutionEngine().build_order({"ticker": "TEST", "side": "long"}, {"allowed": False}, 2)
    assert order["status"] == "blocked"
    assert order["quantity"] == 0


def test_execution_engine_rejects_invalid_signal_and_size():
    engine = ExecutionEngine()
    assert engine.build_order({}, {"allowed": True}, 2)["status"] == "blocked"
    assert engine.build_order({"ticker": "TEST", "side": "hold"}, {"allowed": True}, 2)["status"] == "blocked"
    assert engine.build_order({"ticker": "TEST", "side": "long"}, {"allowed": True}, 0)["status"] == "blocked"
    assert engine.build_order({"ticker": "TEST", "side": "long"}, {"allowed": True}, -1)["status"] == "blocked"


def test_execution_engine_submit_order_retries_then_fails():
    calls = []

    def submitter(order):
        calls.append(order)
        raise RuntimeError("broker down")

    result = ExecutionEngine().submit_order({"ticker": "TEST"}, submitter=submitter, retries=1)
    assert result["status"] == "failed"
    assert result["error"] == "broker down"
    assert len(calls) == 2


def test_execution_engine_submit_order_without_provider_is_simulated():
    assert ExecutionEngine().submit_order({"ticker": "TEST"})["status"] == "simulated"


def test_execution_engine_place_order_returns_explicit_stub_simulation():
    result = ExecutionEngine().place_order({"ticker": "TEST", "side": "long", "quantity": 2, "status": "pending"})
    assert result["status"] == "simulated"
    assert result["provider"] == "stub"
    assert result["order_id"] == "stub-TEST-long-2"


def test_safety_kill_switch_blocks():
    result = evaluate_safety(SafetyPolicy(kill_switch=True), requested_position_size=1)
    assert result["allowed"] is False
    assert "kill_switch" in result["blocked_reasons"]


def test_safety_rejects_invalid_position_size():
    result = evaluate_safety(SafetyPolicy(), requested_position_size=0)
    assert result["allowed"] is False
    assert result["blocked_reasons"] == ["invalid_position_size"]


def test_safety_position_limit_blocks():
    result = evaluate_safety(SafetyPolicy(max_position_size=5), requested_position_size=6)
    assert result["allowed"] is False
    assert "max_position_size" in result["blocked_reasons"]


def test_safety_valid_request_is_allowed():
    result = evaluate_safety(SafetyPolicy(max_position_size=5), requested_position_size=2)
    assert result["allowed"] is True
    assert result["blocked_reasons"] == []


def test_load_risk_check_accepts_csv(tmp_path):
    path = tmp_path / "risk_summary.csv"
    pd.DataFrame([{"n_trades": 10, "win_rate": 0.5, "profit_factor": 1.5, "avg_rr": 1.2, "sharpe": 0.8, "sortino": 1.1, "warning_low_pf": False, "warning_nonpositive_sharpe": False}]).to_csv(path, index=False)
    assert _load_risk_check(str(path))["allowed"] is True


def _write_risk(path):
    pd.DataFrame([{"n_trades": 10, "win_rate": 0.5, "profit_factor": 1.5, "avg_rr": 1.2, "sharpe": 0.8, "sortino": 1.1, "warning_low_pf": False, "warning_nonpositive_sharpe": False, "daily_loss": 0.0, "current_exposure": 0.0}]).to_csv(path, index=False)


def test_run_bot_blocks_on_kill_switch(tmp_path, capsys):
    from scripts import run_bot
    risk = tmp_path / "risk_summary.csv"
    _write_risk(risk)
    old = sys.argv
    sys.argv = ["run_bot.py", "--ticker", "TEST", "--side", "long", "--position-size", "2", "--dry-run", "--kill-switch", "--risk-summary-file", str(risk)]
    try:
        with pytest.raises(SystemExit) as exc:
            run_bot.main()
        assert exc.value.code == 1
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "status: blocked" in out
    assert "kill_switch" in out


def test_run_bot_paper_live_uses_stub_simulation(tmp_path, capsys):
    from scripts import run_bot
    risk = tmp_path / "risk_summary.csv"
    state = tmp_path / "bot_state.json"
    _write_risk(risk)
    old = sys.argv
    sys.argv = ["run_bot.py", "--ticker", "TEST", "--side", "long", "--position-size", "2", "--paper-live", "--risk-summary-file", str(risk), "--state-file", str(state)]
    try:
        run_bot.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "status: simulated" in out
    assert "paper_live: True" in out
    assert json.loads(state.read_text())["last_status"] == "simulated"
