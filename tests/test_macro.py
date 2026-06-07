"""
test_macro.py — Test minimi Fase 10: macro regime classification
"""
import pandas as pd
import numpy as np
import pytest
from src.macro import classify_macro_regime, compute_macro_signals, get_regime_series


def _make_signals(gdp_trend: float, inflation_yoy: float) -> pd.Series:
    return pd.Series({
        "rate": 5.0,
        "inflation": 300.0,
        "gdp": 25000.0,
        "rate_trend": 0.01,
        "inflation_yoy": inflation_yoy,
        "inflation_trend": 0.1,
        "gdp_trend": gdp_trend,
    })


class TestClassifyRegime:
    def test_goldilocks(self):
        row = _make_signals(gdp_trend=0.01, inflation_yoy=2.0)
        assert classify_macro_regime(row) == "goldilocks"

    def test_risk_on(self):
        row = _make_signals(gdp_trend=0.01, inflation_yoy=4.5)
        assert classify_macro_regime(row) == "risk_on"

    def test_risk_off(self):
        row = _make_signals(gdp_trend=-0.01, inflation_yoy=1.5)
        assert classify_macro_regime(row) == "risk_off"

    def test_stagflation(self):
        row = _make_signals(gdp_trend=-0.01, inflation_yoy=5.0)
        assert classify_macro_regime(row) == "stagflation"


class TestComputeSignals:
    def _make_df(self):
        idx = pd.date_range("2010-01-31", periods=24, freq="ME")
        df = pd.DataFrame({
            "rate":      np.linspace(0.1, 5.0, 24),
            "inflation": np.linspace(200, 320, 24),
            "gdp":       np.linspace(14000, 22000, 24),
        }, index=idx)
        return df

    def test_columns_present(self):
        df = compute_macro_signals(self._make_df())
        for col in ["rate_trend", "inflation_yoy", "gdp_trend"]:
            assert col in df.columns

    def test_no_nans_after_dropna(self):
        df = compute_macro_signals(self._make_df())
        assert df[["rate_trend", "inflation_yoy", "gdp_trend"]].isna().sum().sum() == 0

    def test_regime_series_length(self):
        df = compute_macro_signals(self._make_df())
        series = get_regime_series(df)
        assert len(series) == len(df)
        assert set(series.unique()).issubset({"goldilocks", "risk_on", "risk_off", "stagflation"})
