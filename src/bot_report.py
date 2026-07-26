from __future__ import annotations

from typing import Any


def build_bot_report(*, ticker: str, side: str, quantity: int, status: str, dry_run: bool, paper_live: bool, risk_allowed: bool, blocked_reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "status": status,
        "dry_run": dry_run,
        "paper_live": paper_live,
        "risk_allowed": risk_allowed,
        "blocked_reasons": blocked_reasons or [],
    }


def format_bot_report(report: dict[str, Any]) -> str:
    lines = [
        f"ticker: {report.get('ticker')}",
        f"side: {report.get('side')}",
        f"quantity: {report.get('quantity')}",
        f"status: {report.get('status')}",
        f"dry_run: {report.get('dry_run')}",
        f"paper_live: {report.get('paper_live')}",
        f"risk_allowed: {report.get('risk_allowed')}",
    ]
    blocked = report.get("blocked_reasons") or []
    if blocked:
        lines.append(f"blocked_reasons: {','.join(blocked)}")
    return "\n".join(lines)
