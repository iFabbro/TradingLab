"""
run_macro.py — CLI per il regime macro corrente
Uso: python scripts/run_macro.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.macro import (
    compute_macro_signals,
    current_regime_report,
    fetch_macro_series,
    get_regime_series,
)
from src.output_schema import ensure_parent_dir, validate_macro_snapshot


if __name__ == "__main__":
    print("Scaricamento dati FRED...")
    df = fetch_macro_series(start="2010-01-01")
    df_sig = compute_macro_signals(df)
    df_sig["macro_regime"] = get_regime_series(df_sig)

    current_regime_report(df_sig)

    macro_snapshot = (
        df_sig.reset_index()
        .rename(columns={"index": "date"})
        .copy()
    )

    macro_snapshot = validate_macro_snapshot(macro_snapshot)

    out_path = ensure_parent_dir(Path("reports") / "macro_snapshot.csv")
    macro_snapshot.to_csv(out_path, index=False)

    print(f"Salvato: {out_path}")
