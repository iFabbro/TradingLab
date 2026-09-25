"""Reproducible external-data experiment runner.

Usage:
    python experiments/run_experiment.py --symbol spy --start 2015-01-01 --end 2025-01-01

The runner downloads data, stores the exact dataset plus request/provider
metadata, and writes a machine-readable manifest. It does not silently fall
back to synthetic or local data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.providers import StooqCSVProvider, save_dataset, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--provider", default="stooq", choices=["stooq"])
    parser.add_argument("--output-root", default="experiments/results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.provider != "stooq":
        raise ValueError(f"unsupported provider: {args.provider}")

    provider = StooqCSVProvider()
    df = provider.fetch(args.symbol, args.start, args.end, args.interval)
    if df.empty:
        raise RuntimeError("provider returned no market data")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.has_duplicates:
        raise ValueError("provider returned invalid timestamps")
    if not df["close"].map(pd.api.types.is_number).all() or (df["close"] <= 0).any():
        raise ValueError("provider returned invalid close prices")

    run_id = f"{args.symbol.lower()}_{args.start}_{args.end}_{args.interval}_{args.provider}"
    out = Path(args.output_root) / run_id
    dataset_path = out / "dataset.csv"
    metadata_path = out / "metadata.json"
    sha256 = save_dataset(df, dataset_path)
    metadata = {
        "run_id": run_id,
        "symbol": args.symbol.upper(),
        "start_requested": args.start,
        "end_requested": args.end,
        "interval": args.interval,
        "provider": provider.metadata(),
        "rows": int(len(df)),
        "actual_start": df.index.min().isoformat(),
        "actual_end": df.index.max().isoformat(),
        "dataset_sha256": sha256,
        "schema": {"index": "datetime", "columns": list(df.columns)},
        "synthetic_data": False,
        "local_fallback": False,
    }
    save_metadata(metadata, metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
