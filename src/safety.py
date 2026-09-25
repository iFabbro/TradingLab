from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyPolicy:
    kill_switch: bool = False
    daily_loss_limit: float = 0.0
    daily_loss_pct: float = 0.0
    max_exposure: float = 0.0
    max_position_size: int = 0
    paper_live: bool = False
    dry_run: bool = True


def evaluate_safety(
    policy: SafetyPolicy,
    *,
    current_daily_loss: float = 0.0,
    current_exposure: float = 0.0,
    requested_position_size: int = 0,
) -> dict:
    """Evaluate hard execution gates before an order can be built."""
    if requested_position_size <= 0:
        return {"allowed": False, "blocked_reasons": ["invalid_position_size"], "dry_run": policy.dry_run, "paper_live": policy.paper_live}

    blocked_reasons: list[str] = []
    if policy.kill_switch:
        blocked_reasons.append("kill_switch")
    if policy.daily_loss_limit > 0 and current_daily_loss >= policy.daily_loss_limit:
        blocked_reasons.append("daily_loss_limit")
    if policy.daily_loss_pct > 0 and policy.daily_loss_limit <= 0 and current_daily_loss >= policy.daily_loss_pct:
        blocked_reasons.append("daily_loss_pct")
    if policy.max_exposure > 0 and current_exposure > policy.max_exposure:
        blocked_reasons.append("max_exposure")
    if policy.max_position_size > 0 and requested_position_size > policy.max_position_size:
        blocked_reasons.append("max_position_size")

    return {
        "allowed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "dry_run": policy.dry_run,
        "paper_live": policy.paper_live,
    }
