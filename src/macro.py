"""
macro.py — Fase 10: Macro-based strategy
Scarica tassi, inflazione, crescita da FRED e classifica il regime macro.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import pandas_datareader.data as web
    _HAS_PDR = True
except ImportError:
    _HAS_PDR = False

FRED_SERIES = {
    "rate":      "DFF",       # Fed Funds Rate
    "inflation": "CPIAUCSL",  # CPI (YoY calcolato sotto)
    "gdp":       "GDP",       # PIL reale USA (trimestrale)
}

LOOKBACK_MONTHS = 6


def fetch_macro_series(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Scarica le tre serie FRED e le ricampiona mensile.
    Richiede pandas_datareader. Senza rete lancia RuntimeError con istruzioni.
    """
    if not _HAS_PDR:
        raise RuntimeError(
            "Installa pandas_datareader: pip install pandas-datareader"
        )
    frames = {}
    for name, ticker in FRED_SERIES.items():
        s = web.DataReader(ticker, "fred", start=start)[ticker]
        s = s.resample("ME").last().ffill()
        frames[name] = s
    df = pd.DataFrame(frames).dropna(how="all")
    return df


def compute_macro_signals(df: pd.DataFrame, window: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """
    Aggiunge colonne di trend (variazione % su `window` mesi):
      rate_trend, inflation_trend, gdp_trend
    Aggiunge inflation_yoy (CPI % su 12 mesi).
    """
    out = df.copy()
    out["rate_trend"]      = out["rate"].pct_change(window)
    out["inflation_yoy"]   = out["inflation"].pct_change(12) * 100  # in %
    out["inflation_trend"] = out["inflation_yoy"].diff(window)
    # GDP trimestrale → trend su 2 periodi
    out["gdp_trend"]       = out["gdp"].pct_change(2)
    return out.dropna(subset=["rate_trend", "inflation_yoy", "gdp_trend"])


def classify_macro_regime(row: pd.Series) -> str:
    """
    Classifica una singola riga in 4 regimi macro:
      - goldilocks  : crescita alta, inflazione bassa/stabile
      - risk_on     : crescita alta, inflazione alta (early/mid cycle)
      - risk_off    : crescita bassa, inflazione bassa (recessione)
      - stagflation : crescita bassa, inflazione alta
    Soglie conservative e facilmente modificabili.
    """
    growth_up    = row["gdp_trend"] > 0.005      # PIL cresce > 0.5%
    inflation_hi = row["inflation_yoy"] > 3.0    # CPI annuo > 3%

    if growth_up and not inflation_hi:
        return "goldilocks"
    elif growth_up and inflation_hi:
        return "risk_on"
    elif not growth_up and inflation_hi:
        return "stagflation"
    else:
        return "risk_off"


def get_regime_series(df_signals: pd.DataFrame) -> pd.Series:
    """Applica classify_macro_regime su tutto il DataFrame. Ritorna Series."""
    return df_signals.apply(classify_macro_regime, axis=1).rename("macro_regime")


def current_regime_report(df_signals: pd.DataFrame) -> None:
    """Stampa a terminale il regime corrente e i valori chiave."""
    last = df_signals.iloc[-1]
    regime = classify_macro_regime(last)
    print("=" * 40)
    print(f"  DATA           : {df_signals.index[-1].date()}")
    print(f"  REGIME MACRO   : {regime.upper()}")
    print(f"  Fed Rate trend : {last['rate_trend']:+.3f}")
    print(f"  CPI YoY        : {last['inflation_yoy']:+.2f}%")
    print(f"  GDP trend      : {last['gdp_trend']:+.4f}")
    print("=" * 40)
