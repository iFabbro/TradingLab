"""
Loader OHLCV unico — sorgente: yfinance.
Gestisce cache automatica in Parquet.
"""
import yfinance as yf
import pandas as pd
from .schema import normalize
from . import cache as cache_mod

def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    use_cache: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Scarica OHLCV da yfinance con cache Parquet locale.

    Args:
        ticker: es. "AAPL", "BTC-USD"
        start: "YYYY-MM-DD"
        end:   "YYYY-MM-DD"
        interval: "1d", "1h", "1wk", ...
        use_cache: se True, legge dalla cache se disponibile
        force_download: se True, ignora la cache e riscarica

    Returns:
        DataFrame normalizzato con colonne [open, high, low, close, volume]
        e index datetime (name="date").
    """
    if use_cache and not force_download and cache_mod.exists(ticker, interval):
        df = cache_mod.load(ticker, interval)
        return df

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"Nessun dato scaricato per {ticker} ({interval})")

    # yfinance >=0.2 restituisce MultiIndex se multi-ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = normalize(raw)

    if use_cache:
        cache_mod.save(df, ticker, interval)

    return df
