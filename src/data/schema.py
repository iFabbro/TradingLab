"""
Schema standard OHLCV per TradingLab.
Colonne richieste, tipi attesi e funzione di validazione.
"""
import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_map)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[OHLCV_COLUMNS]
    return df.sort_index()

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    return normalize(df)
