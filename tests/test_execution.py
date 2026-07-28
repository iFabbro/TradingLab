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


def test_execution_engine_blocks_when_signal_missing_fields():
    engine = ExecutionEngine()
    order = engine.build_order(signal={}, risk_check={"allowed": True}, position_size=2)

    assert order["status"] == "blocked"
    assert order["quantity"] == 0


def test_execution_engine_submit_order_fails_on_failure():
    engine = ExecutionEngine()
    calls = []
    def submitter(order):
        calls.append(order)
        raise RuntimeError("broker down")
    result = engine.submit_order({"ticker": "TEST", "side": "long", "quantity": 2, "status": "pending"}, submitter=submitter, retries=1)
    assert result["status"] == "failed"
    assert result["error"] == "broker down"
    assert len(calls) == 2


def test_execution_engine_place_order_returns_stub_response():
    engine = ExecutionEngine()
    order = {"ticker": "TEST", "side": "long", "quantity": 2, "status": "pending"}
    result = engine.place_order(order)
    assert result["status"] == "submitted"
    assert result["provider"] == "stub"
    assert result["order_id"] == "stub-TEST-long-2"


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


def test_run_bot_blocks_on_kill_switch(tmp_path, capsys):
    from scripts import run_bot
    import sys

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
                "daily_loss": 0.0,
                "current_exposure": 0.0,
            }
        ]
    ).to_csv(path, index=False)

    old = sys.argv
    sys.argv = [
        "run_bot.py",
        "--ticker",
        "TEST",
        "--side",
        "long",
        "--position-size",
        "2",
        "--dry-run",
        "--kill-switch",
        "--risk-summary-file",
        str(path),
    ]
    try:
        try:
            run_bot.main()
        except SystemExit as exc:
            assert exc.code == 1
    finally:
        sys.argv = old

    out = capsys.readouterr().out
    assert "status: blocked" in out
    assert "kill_switch" in out


def test_run_bot_shows_paper_live_flag(tmp_path, capsys):
    from scripts import run_bot
    import sys

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
                "daily_loss": 0.0,
                "current_exposure": 0.0,
            }
        ]
    ).to_csv(path, index=False)

    old = sys.argv
    sys.argv = [
        "run_bot.py",
        "--ticker",
        "TEST",
        "--side",
        "long",
        "--position-size",
        "2",
        "--paper-live",
        "--risk-summary-file",
        str(path),
    ]
    try:
        run_bot.main()
    finally:
        sys.argv = old

    out = capsys.readouterr().out
    assert "paper_live: True" in out


def test_run_bot_paper_live_executes_order_stub(tmp_path, capsys):
    from scripts import run_bot
    import sys, json
    risk = tmp_path / "risk_summary.csv"
    state = tmp_path / "bot_state.json"
    pd.DataFrame([
        {"n_trades": 10, "win_rate": 0.5, "profit_factor": 1.5, "avg_rr": 1.2, "sharpe": 0.8, "sortino": 1.1, "warning_low_pf": False, "warning_nonpositive_sharpe": False, "daily_loss": 0.0, "current_exposure": 0.0}
    ]).to_csv(risk, index=False)
    old = sys.argv
    sys.argv = ["run_bot.py", "--ticker", "TEST", "--side", "long", "--position-size", "2", "--paper-live", "--risk-summary-file", str(risk), "--state-file", str(state)]
    try:
        run_bot.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "status: submitted" in out
    assert "paper_live: True" in out
    data = json.loads(state.read_text())
    assert data["last_status"] == "submitted"


def test_run_bot_persists_state(tmp_path, capsys):
    from scripts import run_bot
    import sys, json

    risk = tmp_path / "risk_summary.csv"
    state = tmp_path / "bot_state.json"
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
                "daily_loss": 0.0,
                "current_exposure": 0.0,
            }
        ]
    ).to_csv(risk, index=False)

    old = sys.argv
    sys.argv = [
        "run_bot.py",
        "--ticker",
        "TEST",
        "--side",
        "long",
        "--position-size",
        "2",
        "--paper-live",
        "--risk-summary-file",
        str(risk),
        "--state-file",
        str(state),
    ]
    try:
        run_bot.main()
    finally:
        sys.argv = old

    out = capsys.readouterr().out
    assert "status: submitted" in out
    assert state.exists()
    data = json.loads(state.read_text())
    assert data["last_status"] == "submitted"
    assert data["daily_loss_limit"] >= 0
    assert "last_report" in data
