import numpy as np
from typing import Sequence


def run_montecarlo(
    returns: Sequence[float],
    n_simulations: int = 1000,
    initial: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    n = len(r)
    results = np.empty(n_simulations)
    for i in range(n_simulations):
        sampled = rng.choice(r, size=n, replace=True)
        results[i] = initial * np.prod(1 + sampled)
    return results


def montecarlo_summary(
    returns: Sequence[float],
    n_simulations: int = 1000,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
    seed: int | None = 42,
) -> dict:
    sims = run_montecarlo(returns, n_simulations=n_simulations, seed=seed)
    prob_loss_50pct = float(np.mean(sims < 0.5))
    summary = {
        "n_simulations": n_simulations,
        "mean_final_equity": float(np.mean(sims)),
        "std_final_equity": float(np.std(sims)),
        "prob_profit": float(np.mean(sims > 1.0)),
        "prob_loss_50pct": prob_loss_50pct,
        "warning_high_loss_prob": prob_loss_50pct > 0.1,
    }
    for p in percentiles:
        summary[f"p{p}"] = float(np.percentile(sims, p))
    return summary
