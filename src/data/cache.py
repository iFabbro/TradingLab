"""
Cache locale OHLCV con fallback se Parquet non è disponibile.
Path: data/cache/<ticker>_<interval>.{parquet|csv|pkl}
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _base(ticker: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}_{interval}"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _path(ticker: str, interval: str) -> Path:
    base = _base(ticker, interval)
    if _parquet_available():
        return base.with_suffix(".parquet")
    return base.with_suffix(".csv")


def exists(ticker: str, interval: str) -> bool:
    base = _base(ticker, interval)
    return any((base.with_suffix(ext)).exists() for ext in (".parquet", ".csv", ".pkl"))


def cache_exists(ticker: str, interval: str) -> bool:
    return exists(ticker, interval)


def load(ticker: str, interval: str) -> pd.DataFrame:
    base = _base(ticker, interval)
    for path in (base.with_suffix(".parquet"), base.with_suffix(".csv"), base.with_suffix(".pkl")):
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index.name = "date"
                return df
            return pd.read_pickle(path)
    raise FileNotFoundError(f"Cache non trovata: {base}")


def load_cache(ticker: str, interval: str) -> pd.DataFrame:
    return load(ticker, interval)


def save(df: pd.DataFrame, ticker: str, interval: str) -> None:
    path = _path(ticker, interval)
    if path.suffix == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path)


def save_cache(df: pd.DataFrame, ticker: str, interval: str) -> None:
    save(df, ticker, interval)
