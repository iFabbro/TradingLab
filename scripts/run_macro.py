"""
run_macro.py — CLI per il regime macro corrente (Fase 10)
Uso: python scripts/run_macro.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.macro import fetch_macro_series, compute_macro_signals, current_regime_report

if __name__ == "__main__":
    print("Scaricamento dati FRED...")
    df = fetch_macro_series(start="2010-01-01")
    df_sig = compute_macro_signals(df)
    current_regime_report(df_sig)
