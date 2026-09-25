from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ExecutionEngine:
    """Build and submit validated order intents through an injectable provider."""

    def build_order(
        self,
        signal: dict[str, Any],
        risk_check: dict[str, Any],
        position_size: int,
    ) -> dict[str, Any]:
        ticker = signal.get("ticker") if isinstance(signal, dict) else None
        side = signal.get("side") if isinstance(signal, dict) else None
        allowed = isinstance(risk_check, dict) and bool(risk_check.get("allowed", False))

        if not ticker or side not in {"long", "short"} or position_size <= 0 or not allowed:
            return {"ticker": ticker, "side": side, "quantity": 0, "status": "blocked"}

        return {
            "ticker": str(ticker),
            "side": side,
            "quantity": int(position_size),
            "status": "pending",
        }

    def submit_order(
        self,
        order: dict[str, Any],
        submitter: Callable[[dict[str, Any]], Any] | None = None,
        retries: int = 1,
    ) -> dict[str, Any]:
        """Submit an order through an injected callable; never retries forever."""
        if retries < 0:
            raise ValueError("retries must be >= 0")

        if submitter is None:
            return {**order, "status": "simulated"}

        last_error = None
        for _ in range(retries + 1):
            try:
                submitter(order)
                return {**order, "status": "submitted"}
            except Exception as exc:  # provider boundary: convert provider errors to result state
                last_error = str(exc)

        return {**order, "status": "failed", "error": last_error}

    def place_order(self, order: dict[str, Any], provider=None) -> dict[str, Any]:
        """Place through a provider, or return a deterministic simulated result."""
        if provider is None:
            return {
                **order,
                "status": "simulated",
                "provider": "stub",
                "order_id": f"stub-{order.get('ticker')}-{order.get('side')}-{order.get('quantity')}",
            }
        return provider.place_order(order)
