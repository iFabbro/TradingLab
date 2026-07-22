from datetime import datetime, timedelta, timezone

from scripts.tlab_live import current_prices_stale


def test_current_prices_stale_false_for_fresh_rows(tmp_path):
    path = tmp_path / "current_prices.csv"
    path.write_text("ticker,current_price,asof\n", encoding="utf-8")
    fresh_asof = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    rows = [{"ticker": "TEST", "current_price": "100.0", "asof": fresh_asof}]

    assert current_prices_stale(path, rows) is False


def test_current_prices_stale_true_for_stale_rows(tmp_path):
    path = tmp_path / "current_prices.csv"
    path.write_text("ticker,current_price,asof\n", encoding="utf-8")
    stale_asof = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    rows = [{"ticker": "TEST", "current_price": "100.0", "asof": stale_asof}]

    assert current_prices_stale(path, rows) is True


def test_draw_risk_panel_combines_warnings():
    from scripts.tlab_live import draw_risk_panel

    class Dummy:
        def __init__(self):
            self.calls = []
        def getmaxyx(self):
            return (24, 120)
        def addstr(self, *args, **kwargs):
            self.calls.append(args)

    stdscr = Dummy()
    risk = {
        "n_trades": 3,
        "win_rate": 0.0,
        "profit_factor": 1.0,
        "avg_rr": 0.5,
        "sharpe": -0.2,
        "sortino": -0.1,
        "warning_low_pf": True,
        "warning_nonpositive_sharpe": True,
    }

    draw_risk_panel(stdscr, 0, 0, 80, risk)
    assert any("warn pf/sh" in str(call) for call in stdscr.calls)
