from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Strategy Config
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    name: str
    universe: list[str]
    lookback: int                        # periodi usati per calcolo segnale
    rebalance_freq: str = "monthly"      # "daily" | "weekly" | "monthly"
    long_only: bool = True
    top_n: Optional[int] = None          # quanti asset selezionare (None = tutti)
    params: dict = field(default_factory=dict)  # parametri specifici della strategia

    def validate(self) -> None:
        assert len(self.universe) > 0, "universe non può essere vuoto"
        assert self.lookback > 0, "lookback deve essere > 0"
        assert self.rebalance_freq in ("daily", "weekly", "monthly"), \
            f"rebalance_freq non valido: {self.rebalance_freq}"


# ---------------------------------------------------------------------------
# Base Strategy
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Interfaccia comune per tutte le strategie. Il backtest chiamerà generate_signals()."""

    def __init__(self, config: StrategyConfig) -> None:
        config.validate()
        self.config = config

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """
        Input:  prices — DataFrame con date come index, ticker come colonne
        Output: Series con ticker come index, valore float (score/rank/weight)
                oppure 0/1 per long-only binario
        Deve essere implementato dalle sottoclassi.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name!r})"


# ---------------------------------------------------------------------------
# Strategia 1 — Momentum (cross-sectional)
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """
    Seleziona i top_n asset con il rendimento più alto negli ultimi `lookback` periodi.
    Params opzionali in config.params:
      - skip_last: int (default 1) — periodi recenti da escludere (1-month reversal skip)
    """

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        skip = self.config.params.get("skip_last", 1)
        end_idx = len(prices) - skip if skip > 0 else len(prices)
        start_idx = end_idx - self.config.lookback
        if start_idx < 0:
            return pd.Series(0.0, index=prices.columns)

        window = prices.iloc[start_idx:end_idx]
        returns = (window.iloc[-1] / window.iloc[0]) - 1
        scores = returns.fillna(0.0)

        if self.config.top_n is not None:
            threshold = scores.nlargest(self.config.top_n).min()
            scores = (scores >= threshold).astype(float)

        return scores


# ---------------------------------------------------------------------------
# Strategia 2 — Mean Reversion (z-score)
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategy):
    """
    Genera segnali inversamente proporzionali allo z-score del prezzo rispetto
    alla media mobile degli ultimi `lookback` periodi.
    Params opzionali in config.params:
      - z_threshold: float (default 1.0) — soglia minima di z-score per entrare
    """

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if len(prices) < self.config.lookback:
            return pd.Series(0.0, index=prices.columns)

        window = prices.iloc[-self.config.lookback:]
        mean = window.mean()
        std = window.std().replace(0, float("nan"))
        z = (prices.iloc[-1] - mean) / std
        z_threshold = self.config.params.get("z_threshold", 1.0)

        # Segnale: negativo dello z-score (reversion), filtrato per soglia
        signal = -z
        signal[z.abs() < z_threshold] = 0.0
        if self.config.long_only:
            signal[signal < 0] = 0.0

        return signal.fillna(0.0)


# ---------------------------------------------------------------------------
# Strategia 3 — Trend Following (SMA crossover)
# ---------------------------------------------------------------------------

class TrendFollowingStrategy(BaseStrategy):
    """
    Segnale 1 (long) quando SMA_fast > SMA_slow, 0 altrimenti.
    Params obbligatori in config.params:
      - fast: int — periodo SMA veloce (default 20)
      - slow: int — periodo SMA lento (default config.lookback)
    """

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        fast = self.config.params.get("fast", 20)
        slow = self.config.params.get("slow", self.config.lookback)

        if len(prices) < slow:
            return pd.Series(0.0, index=prices.columns)

        sma_fast = prices.iloc[-fast:].mean()
        sma_slow = prices.iloc[-slow:].mean()
        signal = (sma_fast > sma_slow).astype(float)
        return signal
