"""
portfolio.py — Fase 7: Portfolio Construction
Metodi: equal_weight, inverse_volatility, target_volatility
Input: pd.Series o dict di ritorni per asset/strategia
Output: dict {asset: peso}
"""

import numpy as np
import pandas as pd
from typing import Union


def equal_weight(assets: list[str]) -> dict[str, float]:
    """Pesi uguali su tutti gli asset."""
    if not assets:
        raise ValueError("Lista asset vuota.")
    n = len(assets)
    return {a: round(1.0 / n, 6) for a in assets}


def inverse_volatility(
    returns: Union[pd.DataFrame, dict],
    ann_factor: int = 252,
) -> dict[str, float]:
    """
    Peso proporzionale a 1/volatility annualizzata.
    returns: DataFrame (colonne = asset) o dict {asset: pd.Series}
    """
    if isinstance(returns, dict):
        returns = pd.DataFrame(returns)
    if returns.empty:
        raise ValueError("returns vuoto.")

    vols = returns.std() * np.sqrt(ann_factor)
    if (vols == 0).any():
        raise ValueError("Volatilità zero per uno o più asset.")

    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return {a: round(float(w), 6) for a, w in weights.items()}


def target_volatility(
    returns: Union[pd.DataFrame, dict],
    target_vol: float = 0.10,
    ann_factor: int = 252,
) -> dict[str, float]:
    """
    Scala i pesi equal-weight per raggiungere una vol target sul portafoglio.
    Non garantisce che la somma dei pesi sia 1 (può essere <1 se la vol è alta).
    """
    if isinstance(returns, dict):
        returns = pd.DataFrame(returns)
    if returns.empty:
        raise ValueError("returns vuoto.")
    if target_vol <= 0:
        raise ValueError("target_vol deve essere > 0.")

    raw_weights = equal_weight(list(returns.columns))
    w = np.array(list(raw_weights.values()))
    port_returns = returns.values @ w
    port_vol = port_returns.std() * np.sqrt(ann_factor)

    if port_vol == 0:
        raise ValueError("Volatilità portafoglio uguale a zero.")

    scale = target_vol / port_vol
    scaled = w * scale
    return {a: round(float(s), 6) for a, s in zip(returns.columns, scaled)}


def display_weights(weights: dict[str, float]) -> None:
    """Stampa i pesi in formato leggibile da terminale."""
    total = sum(weights.values())
    print(f"\n{'Asset':<20} {'Peso':>10} {'%':>8}")
    print("-" * 42)
    for asset, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"{asset:<20} {w:>10.4f} {w*100:>7.2f}%")
    print("-" * 42)
    print(f"{'TOTALE':<20} {total:>10.4f} {total*100:>7.2f}%")
