"""
Schema standard OHLCV per TradingLab.
Colonne richieste, tipi attesi e funzione di validazione.
"""
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rinomina colonne in lowercase e verifica schema OHLCV."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[OHLCV_COLUMNS]
    return df.sort_index()
