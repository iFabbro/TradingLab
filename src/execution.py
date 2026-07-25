from __future__ import annotations


class ExecutionEngine:
    def build_order(self, signal, risk_check, position_size):
        if not risk_check.get("allowed", False):
            return {"ticker": signal["ticker"], "side": signal["side"], "quantity": 0, "status": "blocked"}
        return {
            "ticker": signal["ticker"],
            "side": signal["side"],
            "quantity": position_size,
            "status": "pending",
        }
