from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

FRED_SERIES = {
    "rate": "DFF",
    "inflation": "CPIAUCSL",
    "gdp": "GDP",
}

FALLBACK_CSV = Path("data/macro/fallback_macro.csv")


def _load_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    last_err = None
    for _ in range(2):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            value_col = [c for c in df.columns if c != "DATE"][0]
            idx = pd.to_datetime(df["DATE"])
            vals = pd.to_numeric(df[value_col], errors="coerce")
            ser = pd.Series(vals.values, index=idx, name=series_id).dropna()
            ser.index = pd.to_datetime(ser.index)
            return ser
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Impossibile scaricare {series_id}: {last_err}")


def _load_fallback_frame() -> pd.DataFrame:
    if not FALLBACK_CSV.exists():
        raise RuntimeError(f"Fallback mancante: {FALLBACK_CSV}")
    df = pd.read_csv(FALLBACK_CSV, parse_dates=["date"]).set_index("date")
    return df.sort_index()


def fetch_macro_series(start: str = "2000-01-01") -> pd.DataFrame:
    try:
        frames = {}
        for name, series_id in FRED_SERIES.items():
            s = _load_fred_series(series_id)
            if series_id == "GDP":
                s = s.resample("ME").ffill()
            else:
                s = s.resample("ME").last().ffill()
            frames[name] = s
        df = pd.DataFrame(frames).sort_index()
        df = df.loc[pd.Timestamp(start):]
        return df.dropna(how="all")
    except Exception:
        df = _load_fallback_frame()
        return df.loc[pd.Timestamp(start):]


def compute_macro_signals(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    out = df.copy()
    out["rate_trend"] = out["rate"].pct_change(window)
    out["inflation_yoy"] = out["inflation"].pct_change(12) * 100
    out["inflation_trend"] = out["inflation_yoy"].diff(window)
    out["gdp_trend"] = out["gdp"].pct_change(2)
    return out.dropna(subset=["rate_trend", "inflation_yoy", "gdp_trend"])


def classify_macro_regime(row: pd.Series) -> str:
    growth_up = row["gdp_trend"] > 0.005
    inflation_hi = row["inflation_yoy"] > 3.0
    if growth_up and not inflation_hi:
        return "goldilocks"
    if growth_up and inflation_hi:
        return "risk_on"
    if not growth_up and inflation_hi:
        return "stagflation"
    return "risk_off"


def get_regime_series(df_signals: pd.DataFrame) -> pd.Series:
    return df_signals.apply(classify_macro_regime, axis=1).rename("macro_regime")


def current_regime_report(df_signals: pd.DataFrame) -> None:
    last = df_signals.iloc[-1]
    regime = classify_macro_regime(last)
    print("=" * 40)
    print(f"DATA           : {df_signals.index[-1].date()}")
    print(f"REGIME MACRO   : {regime.upper()}")
    print(f"Fed Rate trend : {last['rate_trend']:+.3f}")
    print(f"CPI YoY        : {last['inflation_yoy']:+.2f}%")
    print(f"GDP trend      : {last['gdp_trend']:+.4f}")
    print("=" * 40)
