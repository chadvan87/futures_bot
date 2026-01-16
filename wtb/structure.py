from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Pivots:
    highs: List[Tuple[int, float]]
    lows: List[Tuple[int, float]]


def detect_pivots(high: np.ndarray, low: np.ndarray, w: int = 2) -> Pivots:
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    n = len(high)
    if n < 2 * w + 1:
        return Pivots(highs, lows)
    for i in range(w, n - w):
        h = high[i]
        l = low[i]
        if h == np.max(high[i - w:i + w + 1]):
            highs.append((i, float(h)))
        if l == np.min(low[i - w:i + w + 1]):
            lows.append((i, float(l)))
    return Pivots(highs=highs, lows=lows)


@dataclass
class RangeInfo:
    low: float
    high: float
    height: float


def recent_range(high: np.ndarray, low: np.ndarray, lookback: int) -> RangeInfo:
    if len(high) == 0:
        return RangeInfo(low=0.0, high=0.0, height=0.0)
    lb = min(lookback, len(high))
    r_low = float(np.min(low[-lb:]))
    r_high = float(np.max(high[-lb:]))
    return RangeInfo(low=r_low, high=r_high, height=r_high - r_low)


def pick_setup_type(side_mode: str, adx_value: float, atrp: float, corr_btc: float) -> str:
    """Heuristic setup label for algo-only plan (used when LLM overlay is missing).

    - Trendy markets prefer TREND_PULLBACK / BREAKOUT_RETEST
    - Range markets prefer RANGE_SWEEP_RECLAIM / VOLATILITY_FADE
    """
    # Volatility extremes first (works in any regime)
    if atrp >= 8.0:
        return "VOLATILITY_FADE"

    if adx_value >= 25:
        # Trending. If the symbol is extremely BTC-correlated, we tend to prefer
        # breakout/retest behavior (continuation) over mean-reversion labels.
        return "BREAKOUT_RETEST" if corr_btc >= 0.85 else "TREND_PULLBACK"

    # Not trending -> treat as range behavior by default
    return "RANGE_SWEEP_RECLAIM"
