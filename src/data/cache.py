"""
Cache locale OHLCV in formato Parquet.
Path: data/cache/<ticker>_<interval>.parquet
"""
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

def _path(ticker: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}_{interval}.parquet"

def exists(ticker: str, interval: str) -> bool:
    return _path(ticker, interval).exists()

def load(ticker: str, interval: str) -> pd.DataFrame:
    p = _path(ticker, interval)
    if not p.exists():
        raise FileNotFoundError(f"Cache non trovata: {p}")
    return pd.read_parquet(p)

def save(df: pd.DataFrame, ticker: str, interval: str) -> None:
    df.to_parquet(_path(ticker, interval))
