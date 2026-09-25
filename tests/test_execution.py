import json
import sys

import pandas as pd
import pytest

from scripts.run_bot import _load_risk_check
from src.execution import ExecutionEngine


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
    result = ExecutionEngine().submit_order({"ticker": "TEST"})
    assert result["status"] == "simulated"


def test_execution_engine_place_order_returns_explicit_stub_simulation():
    result = ExecutionEngine().place_order({"ticker": "TEST", "side": "long", "quantity": 2, "status": "pending"})
    assert result["status"] == "simulated"
    assert result["provider"] == "stub"
    assert result["order_id"] == "stub-TEST-long-2"


def test_load_risk_check_accepts_csv(tmp_path):
    path = tmp_path / "risk_summary.csv"
    pd.DataFrame([{"n_trades": 10, "win_rate": 0.5, "profit_factor": 1.5, "avg_rr": 1.2, "sharpe": 0.8, "sortino": 1.1, "warning_low_pf": False, "warning_nonpositive_sharpe": False}]).to_csv(path, index=False)
    risk_check = _load_risk_check(str(path))
    assert risk_check["allowed"] is True


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
    data = json.loads(state.read_text())
    assert data["last_status"] == "simulated"
