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

    def submit_order(self, order, submitter=None, retries=1):
        attempts = 0
        last_error = None
        while attempts <= retries:
            try:
                if submitter is None:
                    return {**order, "status": "submitted"}
                submitter(order)
                return {**order, "status": "submitted"}
            except Exception as exc:
                last_error = str(exc)
                attempts += 1
        return {**order, "status": "failed", "error": last_error}

    def place_order(self, order, provider=None):
        if provider is None:
            return {**order, "status": "submitted", "provider": "stub", "order_id": f"stub-{order.get('ticker')}-{order.get('side')}-{order.get('quantity')}"}
        return provider.place_order(order)
