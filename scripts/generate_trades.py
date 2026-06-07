from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.signals import generate_setup


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = {"open", "high", "low", "close"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"Mancano colonne richieste: {sorted(missing)}")
    rename_map = {cols[k]: k for k in required if cols[k] != k}
    if "volume" in cols and cols["volume"] != "volume":
        rename_map[cols["volume"]] = "volume"
    return df.rename(columns=rename_map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trade setups")
    parser.add_argument("--file", required=True, help="CSV con OHLCV")
    parser.add_argument("--ticker", required=True, help="Ticker simbolo")
    parser.add_argument("--direction", choices=["long", "short"], required=True)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--stop-atr-mult", type=float, default=1.5)
    parser.add_argument("--target-rr", type=float, default=2.0)
    args = parser.parse_args()

    df = _load_csv(args.file)
    setup = generate_setup(
        df,
        ticker=args.ticker,
        direction=args.direction,
        atr_window=args.atr_window,
        stop_atr_mult=args.stop_atr_mult,
        target_rr=args.target_rr,
    )

    print(f"ticker: {setup.ticker}")
    print(f"direction: {setup.direction}")
    print(f"regime: {setup.regime}")
    print(f"strategy_tag: {setup.strategy_tag}")
    print(f"entry: {setup.entry:.4f}")
    print(f"stop: {setup.stop:.4f}")
    print(f"target: {setup.target:.4f}")
    print(f"risk_reward: {setup.risk_reward:.2f}")
    print(f"note: {setup.note}")


if __name__ == "__main__":
    main()
