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
