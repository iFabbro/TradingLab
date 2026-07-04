import pandas as pd
import pytest

from src.data.cache import cache_exists, load_cache, save_cache
from src.data.loader import load_ohlcv
from src.data.schema import OHLCV_COLUMNS, normalize_ohlcv

def test_normalize_ohlcv_columns():
    df = pd.DataFrame(
        {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    out = normalize_ohlcv(df)
    assert list(out.columns) == OHLCV_COLUMNS
    assert out.index.name == "date"

def test_normalize_ohlcv_missing_column():
    df = pd.DataFrame(
        {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    with pytest.raises(ValueError):
        normalize_ohlcv(df)

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.cache.CACHE_DIR", tmp_path)
    df = pd.DataFrame(
        {c: [1.0] for c in OHLCV_COLUMNS},
        index=pd.to_datetime(["2024-01-01"]),
    )
    df.index.name = "date"
    save_cache(df, "TEST", "1d")
    assert cache_exists("TEST", "1d")
    out = load_cache("TEST", "1d")
    assert list(out.columns) == OHLCV_COLUMNS

@pytest.mark.integration
def test_load_ohlcv_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.cache.CACHE_DIR", tmp_path)
    out = load_ohlcv("AAPL", "2024-01-01", "2024-01-15", use_cache=True)
    assert not out.empty
    assert list(out.columns) == OHLCV_COLUMNS

@pytest.mark.integration
def test_load_ohlcv_ignores_stale_cache(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setattr("src.data.cache.CACHE_DIR", tmp_path)

    stale = pd.DataFrame(
        {c: [1.0] for c in OHLCV_COLUMNS},
        index=pd.to_datetime(["2024-01-01"]),
    )
    stale.index.name = "date"
    save_cache(stale, "AAPL", "1d")

    cache_file = next(tmp_path.glob("AAPL_1d.*"))
    old_ts = time.time() - 3 * 86400
    os.utime(cache_file, (old_ts, old_ts))

    out = load_ohlcv("AAPL", "2024-01-01", "2024-01-15", use_cache=True)
    assert not out.empty
    assert list(out.columns) == OHLCV_COLUMNS
    assert len(out) != len(stale) or not out.equals(stale)
