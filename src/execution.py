from __future__ import annotations


class ExecutionEngine:
    def build_order(self, signal, risk_check, position_size):
        ticker = signal.get("ticker") if isinstance(signal, dict) else None
        side = signal.get("side") if isinstance(signal, dict) else None
        if not ticker or not side or not risk_check.get("allowed", False):
            return {"ticker": ticker, "side": side, "quantity": 0, "status": "blocked"}
        return {
            "ticker": ticker,
            "side": side,
            "quantity": position_size,
            "status": "pending",
        }
