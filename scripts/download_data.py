#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_ohlcv

def main() -> int:
    parser = argparse.ArgumentParser(description="Download OHLCV data")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = load_ohlcv(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        use_cache=True,
        force_download=args.force,
    )

    print(f"{args.ticker} {args.interval} rows={len(df)}")
    print(df.head().to_string())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
