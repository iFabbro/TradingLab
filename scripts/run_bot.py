from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution import ExecutionEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomous execution bot")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--side", choices=["long", "short"], required=True)
    parser.add_argument("--position-size", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = ExecutionEngine()
    signal = {"ticker": args.ticker, "side": args.side}
    risk_check = {"allowed": True}
    order = engine.build_order(signal=signal, risk_check=risk_check, position_size=args.position_size)

    print(f"ticker: {order['ticker']}")
    print(f"side: {order['side']}")
    print(f"quantity: {order['quantity']}")
    print(f"status: {order['status']}")
    print(f"dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()
