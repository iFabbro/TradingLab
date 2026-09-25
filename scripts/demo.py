"""Safe, deterministic portfolio demo for TradingLab.

This demo exercises the order-building and risk-gating path only. It never
connects to a broker and never submits a real order.
"""

from src.execution import ExecutionEngine


def main() -> None:
    engine = ExecutionEngine()
    signal = {"ticker": "DEMO", "side": "long"}

    allowed = engine.build_order(
        signal=signal,
        risk_check={"allowed": True},
        position_size=2,
    )
    blocked = engine.build_order(
        signal=signal,
        risk_check={"allowed": False},
        position_size=2,
    )

    print("TradingLab safe demo")
    print("---------------------")
    print(f"Allowed order : {allowed}")
    print(f"Blocked order : {blocked}")
    print("No broker connection or real order submission is performed.")


if __name__ == "__main__":
    main()
