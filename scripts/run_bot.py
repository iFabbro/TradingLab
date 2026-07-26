from __future__ import annotations

import argparse
import json
import yaml
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution import ExecutionEngine
from src.safety import SafetyPolicy, evaluate_safety


def _load_risk_check(path: str | None) -> dict:
    if not path:
        return {"allowed": True}
    p = Path(path)
    if p.suffix.lower() == ".csv":
        data = pd.read_csv(p).iloc[0].to_dict()
    else:
        data = json.loads(p.read_text())
    allowed = not bool(data.get("warning_low_pf")) and not bool(data.get("warning_nonpositive_sharpe"))
    data["allowed"] = allowed
    return data


def _load_safety_config(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomous execution bot")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--side", choices=["long", "short"], required=True)
    parser.add_argument("--position-size", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper-live", action="store_true")
    parser.add_argument("--kill-switch", action="store_true")
    parser.add_argument("--max-exposure", type=float, default=0.0)
    parser.add_argument("--max-position-size", type=int, default=0)
    parser.add_argument("--daily-loss-limit", type=float, default=0.0)
    parser.add_argument("--safety-config", default="config/safety.yaml")
    parser.add_argument("--risk-summary-file", default=None)
    args = parser.parse_args()

    engine = ExecutionEngine()
    signal = {"ticker": args.ticker, "side": args.side}
    risk_check = _load_risk_check(args.risk_summary_file)
    safety_cfg = _load_safety_config(args.safety_config)
    policy = SafetyPolicy(
        kill_switch=bool(safety_cfg.get("kill_switch", False) or args.kill_switch),
        daily_loss_limit=float(safety_cfg.get("daily_loss_limit", 0.0) or args.daily_loss_limit),
        max_exposure=float(safety_cfg.get("max_exposure", 0.0) or args.max_exposure),
        max_position_size=int(safety_cfg.get("max_position_size", 0) or args.max_position_size),
        paper_live=bool(safety_cfg.get("paper_live", False) or args.paper_live),
        dry_run=bool(safety_cfg.get("dry_run", True) or args.dry_run),
    )
    safety = evaluate_safety(
        policy,
        current_daily_loss=float(risk_check.get("daily_loss", 0.0)),
        current_exposure=float(risk_check.get("current_exposure", 0.0)),
        requested_position_size=args.position_size,
    )
    if not safety["allowed"]:
        print("status: blocked")
        print(f"blocked_reasons: {','.join(safety['blocked_reasons'])}")
        print(f"dry_run: {safety['dry_run']}")
        print(f"paper_live: {safety['paper_live']}")
        print(f"risk_allowed: {risk_check['allowed']}")
        raise SystemExit(1)

    risk_check["allowed"] = bool(risk_check.get("allowed", False)) and safety["allowed"]
    order = engine.build_order(signal=signal, risk_check=risk_check, position_size=args.position_size)

    print(f"ticker: {order['ticker']}")
    print(f"side: {order['side']}")
    print(f"quantity: {order['quantity']}")
    print(f"status: {order['status']}")
    print(f"dry_run: {safety['dry_run']}")
    print(f"paper_live: {safety['paper_live']}")
    print(f"risk_allowed: {risk_check['allowed']}")


if __name__ == "__main__":
    main()
