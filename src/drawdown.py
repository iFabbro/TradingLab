import numpy as np
from typing import Sequence


def equity_curve(returns: Sequence[float], initial: float = 1.0) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return initial * np.cumprod(1 + r)


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak


def max_drawdown(returns: Sequence[float]) -> float:
    eq = equity_curve(returns)
    dd = drawdown_series(eq)
    return float(np.min(dd))


def max_drawdown_duration(returns: Sequence[float]) -> int:
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak
    max_dur = 0
    cur_dur = 0
    for u in underwater:
        if u:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def calmar_ratio(returns: Sequence[float]) -> float:
    ann_return = float(np.mean(returns)) * len(returns)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("inf")
    return ann_return / mdd


def drawdown_periods(returns: Sequence[float]) -> list[dict]:
    """
    Restituisce la lista di tutti i drawdown con:
    - start: indice di inizio (primo bar sotto il peak)
    - end: indice di fine del drawdown (recovery o fine serie)
    - trough: indice del punto di massima perdita
    - depth: profondità (valore negativo)
    - duration: durata in barre (inizio -> trough)
    - recovery: barre per il recovery completo (trough -> fine DD), None se non recuperato
    """
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    dd = drawdown_series(eq)
    n = len(eq)

    periods = []
    in_dd = False
    start = 0
    trough_idx = 0

    for i in range(n):
        if not in_dd and dd[i] < 0:
            in_dd = True
            start = i
            trough_idx = i
        elif in_dd:
            if dd[i] < dd[trough_idx]:
                trough_idx = i
            if dd[i] >= 0:
                periods.append({
                    "start": start,
                    "trough": trough_idx,
                    "end": i,
                    "depth": float(dd[trough_idx]),
                    "duration": trough_idx - start + 1,
                    "recovery": i - trough_idx,
                })
                in_dd = False

    # drawdown non ancora recuperato a fine serie
    if in_dd:
        periods.append({
            "start": start,
            "trough": trough_idx,
            "end": n - 1,
            "depth": float(dd[trough_idx]),
            "duration": trough_idx - start + 1,
            "recovery": None,
        })

    return periods


def max_recovery_time(returns: Sequence[float]) -> int | None:
    """
    Tempo massimo di recovery (in barre) tra tutti i drawdown completati.
    Restituisce None se nessun drawdown è stato completato.
    """
    periods = drawdown_periods(returns)
    completed = [p["recovery"] for p in periods if p["recovery"] is not None]
    return max(completed) if completed else None


def drawdown_suggestions(returns: Sequence[float]) -> list[str]:
    """
    Restituisce suggerimenti operativi concreti basati sull'analisi del drawdown.
    """
    summary = drawdown_summary(returns)
    periods = drawdown_periods(returns)
    suggestions = []

    mdd = summary["max_drawdown"]
    max_dur = summary["max_drawdown_duration"]
    calmar = summary["calmar_ratio"]
    completed = [p for p in periods if p["recovery"] is not None]
    avg_recovery = np.mean([p["recovery"] for p in completed]) if completed else None

    if mdd < -0.20:
        suggestions.append(
            f"Max drawdown {mdd:.1%}: supera il 20%. Valuta position sizing più conservativo "
            "o un trailing stop per ridurre l'esposizione nelle fasi di perdita prolungata."
        )
    elif mdd < -0.10:
        suggestions.append(
            f"Max drawdown {mdd:.1%}: nella fascia 10-20%. Monitora la frequenza dei DD "
            "e considera un limite di perdita giornaliero/settimanale."
        )
    else:
        suggestions.append(f"Max drawdown {mdd:.1%}: contenuto sotto il 10%. Profilo di rischio accettabile.")

    if max_dur > 20:
        suggestions.append(
            f"Durata massima drawdown: {max_dur} barre. Un DD così prolungato indica periodi "
            "di stagnazione; valuta filtri di regime o pausa operativa dopo N barre consecutive in perdita."
        )

    if calmar < 0.5:
        suggestions.append(
            f"Calmar ratio {calmar:.2f}: basso. Il rendimento non compensa adeguatamente il DD. "
            "Rivedi il rapporto risk/reward o la frequenza dei trade."
        )
    elif calmar > 2.0:
        suggestions.append(f"Calmar ratio {calmar:.2f}: ottimo. La strategia compensa bene il rischio di drawdown.")

    if avg_recovery is not None and avg_recovery > 10:
        suggestions.append(
            f"Tempo medio di recovery: {avg_recovery:.1f} barre. Recovery lenti suggeriscono "
            "trade che perdono momentum rapidamente; considera stop loss più stretti o riduzione del size dopo una perdita."
        )

    unrecovered = [p for p in periods if p["recovery"] is None]
    if unrecovered:
        suggestions.append(
            "Drawdown ancora aperto a fine serie: la strategia non ha ancora recuperato il picco. "
            "Considera se il backtest copre un periodo sufficientemente lungo."
        )

    if not suggestions:
        suggestions.append("Nessuna criticità rilevata nei drawdown analizzati.")

    return suggestions


def drawdown_summary(returns: Sequence[float]) -> dict:
    eq = equity_curve(returns)
    dd = drawdown_series(eq)
    return {
        "max_drawdown": float(np.min(dd)),
        "avg_drawdown": float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0,
        "max_drawdown_duration": max_drawdown_duration(returns),
        "calmar_ratio": calmar_ratio(returns),
        "final_equity": float(eq[-1]) if len(eq) > 0 else 1.0,
    }
