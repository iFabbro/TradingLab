"""External market-data providers for reproducible experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    name: str
    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame: ...
    def metadata(self) -> dict: ...


@dataclass
class YahooFinanceProvider:
    """Yahoo Finance adapter backed by the installed yfinance package."""
    name: str = "yahoo_finance"
    auto_adjust: bool = True
    repair: bool = False

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("YahooFinanceProvider currently supports interval='1d' only")
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            repair=self.repair,
            actions=False,
            progress=False,
            threads=False,
            ignore_tz=True,
            multi_level_index=False,
            timeout=30,
        )
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError(f"Yahoo Finance returned no usable Close data for {symbol}")
        df = df.rename(columns={"Close": "close"})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["close"]].sort_index().astype(float)

    def metadata(self) -> dict:
        import yfinance as yf
        return {
            "provider": self.name,
            "library": "yfinance",
            "library_version": yf.__version__,
            "auto_adjust": self.auto_adjust,
            "repair": self.repair,
            "interval": "1d",
        }


@dataclass
class StooqCSVProvider:
    """Legacy Stooq CSV adapter retained as an explicit provider option."""
    name: str = "stooq"
    base_url: str = "https://stooq.com/q/d/l/"

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("StooqCSVProvider currently supports interval='1d' only")
        import urllib.request
        from io import StringIO
        query = f"?s={symbol.lower()}&d1={start.replace('-', '')}&d2={end.replace('-', '')}&i=d"
        with urllib.request.urlopen(self.base_url + query, timeout=30) as response:
            raw = response.read().decode("utf-8")
        df = pd.read_csv(StringIO(raw))
        if "Date" not in df or "Close" not in df:
            raise ValueError("provider response does not contain Date/Close")
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        df = df.rename(columns={"Date": "date", "Close": "close"}).set_index("date").sort_index()
        return df[["close"]].astype(float)

    def metadata(self) -> dict:
        return {"provider": self.name, "base_url": self.base_url}


def save_dataset(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv = df.to_csv(index=True)
    path.write_text(csv, encoding="utf-8")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def save_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
