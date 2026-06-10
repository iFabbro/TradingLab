from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OPEN_TRADES = ROOT / "data" / "trades" / "open_trades.csv"
OUTFILE = ROOT / "data" / "live" / "current_prices.csv"


def load_open_tickers(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    tickers = []
    for row in rows:
        t = str(row.get("ticker", "")).strip().upper()
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def fetch_last_close(ticker: str) -> float | None:
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        value = df["Close"].dropna()
        if value.empty:
            return None
        return float(value.iloc[-1])
    except Exception:
        return None


def main() -> None:
    tickers = load_open_tickers(OPEN_TRADES)
    now = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, str]] = []
    for ticker in tickers:
        price = fetch_last_close(ticker)
        rows.append(
            {
                "ticker": ticker,
                "current_price": "" if price is None else f"{price:.6f}",
                "asof": now,
            }
        )

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTFILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "current_price", "asof"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
